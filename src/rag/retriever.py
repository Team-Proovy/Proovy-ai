from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agents.workflows.subgraphs.step3_rag import config
from agents.workflows.subgraphs.step3_rag.types import RetrievedDoc
from rag.vector_store.base import BaseVectorStore
from rag.vector_store.mock import MockVectorStore
from rag.vector_store.pgvector import PgVectorStore

logger = logging.getLogger(__name__)


def _create_vector_store() -> BaseVectorStore:
    backend = os.getenv("RAG_VECTORSTORE", "mock").lower()
    if backend == "pgvector":
        dsn = os.getenv("PGVECTOR_DSN", "")
        return PgVectorStore(dsn=dsn)
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
    if top_k is None:
        top_k = config.TOP_K
    top_k = _normalize_top_k(top_k)
    joined = " ".join([query or ""] + [text or "" for text in context_texts]).strip()
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

