"""Vector store base interface used by retriever."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class BaseVectorStore:
    """Minimal vector store interface for our RAG flow.

    Implementations should return documents with keys:
    - id, title, text, score, metadata
    The score is expected to be a float in a consistent scale (e.g. 0.0 - 1.0).
    """

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return list of docs sorted by relevance (desc)."""
        raise NotImplementedError

