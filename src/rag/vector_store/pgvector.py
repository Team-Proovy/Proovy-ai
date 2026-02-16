from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from psycopg import sql

from rag.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector 기반 벡터 스토어.

    코사인 유사도를 사용하여 문서를 검색합니다.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        table_name: str = "document_embeddings",
        pool_size: int = 5,
    ) -> None:
        """PgVectorStore 초기화.

        Args:
            dsn: PostgreSQL 연결 문자열 (기본값: PGVECTOR_DSN 환경변수)
            table_name: 임베딩이 저장된 테이블명
            pool_size: 커넥션 풀 크기
        """
        self.dsn = dsn or os.getenv("PGVECTOR_DSN", "")
        self.table_name = table_name
        self.pool_size = pool_size
        self._pool = None

        if not self.dsn:
            raise ValueError(
                "PGVECTOR_DSN environment variable or dsn parameter is required"
            )

    def _get_pool(self):
        """커넥션 풀을 lazy하게 초기화."""
        if self._pool is None:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError:
                raise ImportError(
                    "psycopg[pool] is required. Install with: pip install 'psycopg[binary,pool]'"
                )

            self._pool = ConnectionPool(
                self.dsn,
                min_size=1,
                max_size=self.pool_size,
                open=True,
            )
            logger.info("PgVectorStore connection pool initialized (size=%d)", self.pool_size)

        return self._pool

    @contextmanager
    def _get_connection(self) -> Generator:
        """커넥션 풀에서 커넥션을 가져옴."""
        pool = self._get_pool()
        with pool.connection() as conn:
            # pgvector extension 등록
            try:
                from pgvector.psycopg import register_vector
                register_vector(conn)
            except ImportError:
                logger.warning("pgvector package not installed, using raw arrays")
            yield conn

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        *,
        batch_size: int = 100,
    ) -> None:
        """문서를 임베딩하여 DB에 저장.

        Args:
            documents: 저장할 문서 리스트. 각 문서는 다음 키를 포함:
                - id 또는 doc_id: 문서 고유 ID
                - title: 문서 제목 (선택)
                - text 또는 content: 문서 내용
                - metadata: 추가 메타데이터 (선택)
            batch_size: 배치 크기
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # 임베딩 함수 import (순환 참조 방지)
        from rag.embeddings import embed_texts

        # 문서 전처리
        processed_docs = []
        texts_to_embed = []

        for doc in documents:
            doc_id = str(doc.get("doc_id") or doc.get("id") or "")
            if not doc_id:
                logger.warning("Skipping document without id: %s", doc)
                continue

            title = str(doc.get("title") or "")
            content = str(doc.get("text") or doc.get("content") or "")
            metadata = doc.get("metadata") or {}

            if not content:
                logger.warning("Skipping document without content: doc_id=%s", doc_id)
                continue

            processed_docs.append({
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "metadata": metadata,
            })
            # 제목과 내용을 합쳐서 임베딩
            texts_to_embed.append(f"{title}\n\n{content}" if title else content)

        if not processed_docs:
            logger.warning("No valid documents to add after preprocessing")
            return

        # 임베딩 생성
        logger.info("Embedding %d documents...", len(processed_docs))
        embeddings = embed_texts(texts_to_embed, batch_size=batch_size)

        # DB에 저장
        insert_query = sql.SQL("""
            INSERT INTO {table}
                (doc_id, title, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id)
            DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
        """).format(table=sql.Identifier(self.table_name))

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(processed_docs), batch_size):
                    batch_docs = processed_docs[i:i + batch_size]
                    batch_embeddings = embeddings[i:i + batch_size]

                    for doc, embedding in zip(batch_docs, batch_embeddings):
                        cur.execute(
                            insert_query,
                            (
                                doc["doc_id"],
                                doc["title"],
                                doc["content"],
                                embedding,
                                json.dumps(doc["metadata"]),
                            ),
                        )

                    conn.commit()
                    logger.debug(
                        "Inserted/updated batch %d-%d of %d documents",
                        i,
                        min(i + batch_size, len(processed_docs)),
                        len(processed_docs),
                    )

        logger.info("Successfully added %d documents to %s", len(processed_docs), self.table_name)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """쿼리와 유사한 문서를 검색.

        코사인 유사도를 사용하여 가장 유사한 문서를 반환합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수
            filters: 메타데이터 필터 (예: {"source": "math"})

        Returns:
            유사도 순으로 정렬된 문서 리스트. 각 문서는 다음 키를 포함:
                - id: 문서 고유 ID
                - title: 문서 제목
                - text: 문서 내용
                - score: 유사도 점수 (0.0 ~ 1.0)
                - metadata: 추가 메타데이터
        """
        if not query or not query.strip():
            return []

        # 쿼리 임베딩
        from rag.embeddings import embed_query
        query_embedding = embed_query(query.strip())

        if not query_embedding:
            logger.warning("Failed to embed query")
            return []

        # 필터 조건 생성
        filter_clause = sql.SQL("")
        filter_params: List[Any] = []

        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append("metadata->>%s = %s")
                filter_params.extend([key, str(value)])

            if filter_conditions:
                filter_clause = sql.SQL("WHERE ") + sql.SQL(" AND ").join(
                    sql.SQL(cond) for cond in filter_conditions
                )

        # 코사인 유사도 검색 (1 - cosine_distance = cosine_similarity)
        # pgvector의 <=> 연산자는 코사인 거리를 반환 (0 = 동일, 2 = 반대)
        search_query = sql.SQL("""
            SELECT
                doc_id,
                title,
                content,
                1 - (embedding <=> %s) AS score,
                metadata
            FROM {table}
            {filter_clause}
            ORDER BY embedding <=> %s
            LIMIT %s
        """).format(
            table=sql.Identifier(self.table_name),
            filter_clause=filter_clause,
        )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                params = [query_embedding] + filter_params + [query_embedding, top_k]
                cur.execute(search_query, params)

                results = []
                for row in cur.fetchall():
                    doc_id, title, content, score, metadata = row
                    results.append({
                        "id": doc_id,
                        "title": title or "",
                        "text": content or "",
                        "score": float(score) if score is not None else 0.0,
                        "metadata": metadata if isinstance(metadata, dict) else {},
                    })

        logger.info(
            "Search completed: query_len=%d, results=%d, top_score=%.3f",
            len(query),
            len(results),
            results[0]["score"] if results else 0.0,
        )

        return results

    def delete_document(self, doc_id: str) -> bool:
        """특정 문서 삭제.

        Args:
            doc_id: 삭제할 문서 ID

        Returns:
            삭제 성공 여부
        """
        delete_query = sql.SQL("DELETE FROM {table} WHERE doc_id = %s").format(
            table=sql.Identifier(self.table_name)
        )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_query, (doc_id,))
                deleted = cur.rowcount > 0
                conn.commit()

        if deleted:
            logger.info("Deleted document: %s", doc_id)
        else:
            logger.warning("Document not found for deletion: %s", doc_id)

        return deleted

    def clear(self) -> int:
        """모든 문서 삭제.

        Returns:
            삭제된 문서 수
        """
        delete_query = sql.SQL("DELETE FROM {table}").format(
            table=sql.Identifier(self.table_name)
        )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_query)
                deleted_count = cur.rowcount
                conn.commit()

        logger.info("Cleared %d documents from %s", deleted_count, self.table_name)
        return deleted_count

    def count(self) -> int:
        """저장된 문서 수 반환."""
        count_query = sql.SQL("SELECT COUNT(*) FROM {table}").format(
            table=sql.Identifier(self.table_name)
        )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(count_query)
                result = cur.fetchone()
                return result[0] if result else 0

    def close(self) -> None:
        """커넥션 풀 종료."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            logger.info("PgVectorStore connection pool closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
