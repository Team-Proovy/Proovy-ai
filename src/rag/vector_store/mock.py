from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.vector_store.base import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    """Deterministic mock vector store for unit/integration tests."""

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        return

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        safe_query = (query or "").strip()
        if not safe_query:
            return []
        base_text = safe_query[:280]
        docs = [
            {
                "id": "mock_doc_1",
                "title": "Mock Document 1",
                "text": base_text,
                "score": 0.85,
                "metadata": {"source": "mock", "chunk_id": "mock_0001"},
            },
            {
                "id": "mock_doc_2",
                "title": "Mock Document 2",
                "text": base_text,
                "score": 0.78,
                "metadata": {"source": "mock", "chunk_id": "mock_0002"},
            },
            {
                "id": "mock_doc_3",
                "title": "Mock Document 3",
                "text": base_text,
                "score": 0.72,
                "metadata": {"source": "mock", "chunk_id": "mock_0003"},
            },
        ]
        if top_k <= 0:
            return []
        return docs[: min(len(docs), top_k)]

