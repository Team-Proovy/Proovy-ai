from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from core.settings import settings
from rag.vector_store.base import BaseVectorStore
from rag.vector_store.mock import MockVectorStore
from rag.vector_store.pgvector import PgVectorStore

logger = logging.getLogger(__name__)


class RetrievedDoc(TypedDict):
    id: str
    title: str
    text: str
    score: Optional[float]
    metadata: Dict[str, Any]


def _create_vector_store() -> BaseVectorStore:
    backend = os.getenv("RAG_VECTORSTORE", "mock").lower()
    if backend == "pgvector":
        # PGVECTOR_DSN이 직접 있으면 최우선 사용, 없으면 DB_* 설정들로 조합
        dsn = os.getenv("PGVECTOR_DSN")
        if not dsn:
            try:
                dsn = settings.POSTGRES_URI
            except Exception:
                logger.warning("Failed to build DSN from settings, falling back to empty")
                dsn = ""
        
        index_name = os.getenv("PGVECTOR_INDEX_NAME", "documents")
        distance_metric = os.getenv("PGVECTOR_DISTANCE", "cosine")
        try:
            # OpenAI 표준인 1536을 기본값으로 사용
            embedding_dim = int(os.getenv("PGVECTOR_EMBEDDING_DIM", "1536"))
        except ValueError:
            embedding_dim = 1536
        return PgVectorStore(
            dsn=dsn,
            index_name=index_name,
            embedding_dim=embedding_dim,
            distance_metric=distance_metric,
        )
    return MockVectorStore()


_VECTOR_STORE: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = _create_vector_store()
    return _VECTOR_STORE


def set_vector_store(store: BaseVectorStore) -> None:
    global _VECTOR_STORE
    _VECTOR_STORE = store


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_top_k(value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, normalized)


def _validate_doc_shape(doc: Dict[str, Any]) -> RetrievedDoc:
    metadata = doc.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "id": str(doc.get("id") or ""),
        "title": str(doc.get("title") or "Untitled"),
        "text": str(doc.get("text") or doc.get("content") or ""),
        "score": _coerce_score(doc.get("score")),
        "metadata": metadata,
    }


def search(
    query: str,
    *,
    context_texts: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RetrievedDoc]:
    """High-level retriever used by graph nodes."""
    if context_texts is None:
        context_texts = []
    
    # agents.config 의존성 대신 환경변수 직접 참조
    if top_k is None:
        try:
            top_k = int(os.getenv("RAG_TOP_K", "3"))
        except ValueError:
            top_k = 3
            
    top_k = _normalize_top_k(top_k)
    
    # 중복 텍스트 합치기 방지
    query_part = query or ""
    additional_parts = []
    for text in (context_texts or []):
        if text and text not in query_part:
            additional_parts.append(text)
            
    joined = " ".join([query_part] + additional_parts).strip()
    
    if not joined:
        return []
    try:
        raw_docs = get_vector_store().search(joined, top_k=top_k, filters=filters)
    except Exception:
        logger.exception(
            "retriever search failed (len_query=%d top_k=%s backend=%s)",
            len(joined),
            top_k,
            os.getenv("RAG_VECTORSTORE", "mock"),
        )
        return []
    if not isinstance(raw_docs, list):
        logger.warning("retriever search returned non-list docs")
        return []
    return [_validate_doc_shape(doc) for doc in raw_docs if isinstance(doc, dict)]

