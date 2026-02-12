from __future__ import annotations

import logging
import os
import re
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


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣]+")


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token.strip()) >= 2
    }


def _coerce_weight(value: str, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def _hybrid_rerank(query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
    """Generic rerank: vector score + lexical overlap (no subject hard-coding)."""
    if not docs:
        return docs
    if os.getenv("RAG_HYBRID_RERANK", "1").lower() not in {"1", "true", "yes"}:
        return docs

    vector_weight = _coerce_weight(os.getenv("RAG_VECTOR_WEIGHT", "0.75"), 0.75)
    lexical_weight = _coerce_weight(os.getenv("RAG_LEXICAL_WEIGHT", "0.25"), 0.25)
    if vector_weight + lexical_weight <= 0:
        return docs

    q_tokens = _tokenize(query)
    reranked: List[RetrievedDoc] = []
    for doc in docs:
        text = str(doc.get("text") or "")
        title = str(doc.get("title") or "")
        d_tokens = _tokenize(f"{title} {text}")
        lexical_score = 0.0
        if q_tokens and d_tokens:
            lexical_score = len(q_tokens & d_tokens) / max(len(q_tokens), 1)

        vector_score = _coerce_score(doc.get("score")) or 0.0
        hybrid_score = (vector_weight * vector_score) + (lexical_weight * lexical_score)

        merged: RetrievedDoc = dict(doc)  # type: ignore[assignment]
        meta = merged.get("metadata")
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["hybrid_vector_score"] = vector_score
        meta["hybrid_lexical_score"] = lexical_score
        merged["metadata"] = meta
        merged["score"] = hybrid_score
        reranked.append(merged)

    reranked.sort(key=lambda item: _coerce_score(item.get("score")) or 0.0, reverse=True)
    return reranked


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
    normalized = [_validate_doc_shape(doc) for doc in raw_docs if isinstance(doc, dict)]
    reranked = _hybrid_rerank(joined, normalized)
    return reranked[:top_k]

