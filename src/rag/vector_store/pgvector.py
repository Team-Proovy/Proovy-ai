from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.vector_store.base import BaseVectorStore


class PgVectorStore(BaseVectorStore):
    """pgvector-backed vector store skeleton.

    TODO: implement connection, index management, and similarity search.
    """

    def __init__(self, dsn: str, index_name: str = "documents") -> None:
        self.dsn = dsn
        self.index_name = index_name

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("pgvector add_documents is not implemented yet")

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("pgvector search is not implemented yet")

