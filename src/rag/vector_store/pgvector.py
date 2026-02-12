from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from rag.embeddings import embed_query, embed_texts
from rag.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PgVectorStore(BaseVectorStore):
    """pgvector-backed vector store."""

    _SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        dsn: str,
        index_name: str = "documents",
        *,
        embedding_dim: int = 3,
        distance_metric: str = "cosine",
    ) -> None:
        if not dsn.strip():
            raise ValueError("PGVECTOR_DSN is required when using pgvector backend")
        if not self._SAFE_NAME.match(index_name):
            raise ValueError(
                "index_name must match ^[A-Za-z_][A-Za-z0-9_]*$ for SQL safety"
            )
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        normalized_metric = distance_metric.strip().lower()
        if normalized_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("distance_metric must be one of: cosine, l2, ip")

        self.dsn = dsn
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.distance_metric = normalized_metric

        self._table_ident = sql.Identifier(self.index_name)
        self._meta_index_ident = sql.Identifier(f"{self.index_name}_metadata_gin_idx")
        self._embedding_index_ident = sql.Identifier(
            f"{self.index_name}_embedding_hnsw_idx"
        )

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        vector_ops = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }[self.distance_metric]
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {table} (
                            id TEXT PRIMARY KEY,
                            title TEXT NOT NULL,
                            text TEXT NOT NULL,
                            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                            embedding VECTOR({dim}) NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    ).format(table=self._table_ident, dim=sql.Literal(self.embedding_dim))
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {meta_idx}
                        ON {table} USING GIN (metadata)
                        """
                    ).format(meta_idx=self._meta_index_ident, table=self._table_ident)
                )
                # HNSW 인덱스로 변경 (소규모~대규모 모두 적합)
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {emb_idx}
                        ON {table} USING hnsw (embedding {ops})
                        """
                    ).format(
                        emb_idx=self._embedding_index_ident,
                        table=self._table_ident,
                        ops=sql.SQL(vector_ops),
                    )
                )
            conn.commit()

    @staticmethod
    def _to_vector_literal(values: List[float]) -> str:
        # 벡터 정규화 (유사도 계산 정확도 향상)
        import math
        norm = sum(v*v for v in values) ** 0.5
        if norm > 0:
            values = [v/norm for v in values]
        return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"

    def _coerce_embedding(self, embedding: Any) -> List[float]:
        if not isinstance(embedding, list):
            raise ValueError("embedding must be a list")
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}"
            )
        return [float(v) for v in embedding]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return

        rows: list[tuple[str, str, str, str, str]] = []
        for idx, doc in enumerate(documents):
            doc_id = str(doc.get("id") or f"doc_{idx + 1}")
            title = str(doc.get("title") or "Untitled")
            text = str(doc.get("text") or doc.get("content") or "")
            metadata = doc.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            
            embedding = doc.get("embedding")
            if embedding is None:
                # 제목과 본문을 합쳐서 임베딩 (검색 품질 대폭 향상)
                combined_text = f"[{title}] {text}"
                embedding = embed_texts([combined_text])[0]
            
            embedding_values = self._coerce_embedding(embedding)
            rows.append(
                (
                    doc_id,
                    title,
                    text,
                    json.dumps(metadata, ensure_ascii=False),
                    self._to_vector_literal(embedding_values),
                )
            )

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {table} (id, title, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (id) DO UPDATE
                        SET
                            title = EXCLUDED.title,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """
                    ).format(table=self._table_ident),
                    rows,
                )
            conn.commit()

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        safe_query = (query or "").strip()
        if not safe_query:
            return []
        q_embedding = embed_query(safe_query)
        q_embedding = self._coerce_embedding(q_embedding)
        q_literal = self._to_vector_literal(q_embedding)

        if self.distance_metric == "cosine":
            distance_expr = sql.SQL("embedding <=> %s::vector")
            score_expr = sql.SQL("(1 - (embedding <=> %s::vector))")
        elif self.distance_metric == "l2":
            distance_expr = sql.SQL("embedding <-> %s::vector")
            score_expr = sql.SQL("(1 / (1 + (embedding <-> %s::vector)))")
        else:  # ip
            distance_expr = sql.SQL("embedding <#> %s::vector")
            score_expr = sql.SQL("(-(embedding <#> %s::vector))")

        # 쿼리 파라미터를 명확하게 관리
        params: list[Any] = []
        # score_expr용
        params.append(q_literal)
        
        where_clause = sql.SQL("")
        if filters and isinstance(filters, dict):
            where_clause = sql.SQL("WHERE metadata @> %s::jsonb")
            params.append(json.dumps(filters, ensure_ascii=False))

        # distance_expr용
        params.append(q_literal)
        # LIMIT용
        params.append(top_k)

        query_sql = sql.SQL(
            """
            SELECT
                id,
                title,
                text,
                metadata,
                {score} AS score
            FROM {table}
            {where}
            ORDER BY {distance} ASC
            LIMIT %s
            """
        ).format(
            score=score_expr,
            table=self._table_ident,
            where=where_clause,
            distance=distance_expr,
        )

        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": str(row.get("id") or ""),
                    "title": str(row.get("title") or "Untitled"),
                    "text": str(row.get("text") or ""),
                    "score": float(row["score"]) if row.get("score") is not None else None,
                    "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                }
            )
        logger.debug("pgvector search complete: top_k=%s count=%s", top_k, len(results))
        return results

