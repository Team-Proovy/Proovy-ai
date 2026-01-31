from __future__ import annotations

from rag.vector_store.mock import MockVectorStore


def test_mock_search_empty_query_returns_empty() -> None:
    store = MockVectorStore()
    assert store.search("", top_k=3) == []


def test_mock_search_respects_top_k_and_shape() -> None:
    store = MockVectorStore()
    docs = store.search("test query", top_k=1)
    assert len(docs) == 1
    doc = docs[0]
    assert set(doc.keys()) >= {"id", "title", "text", "score", "metadata"}

