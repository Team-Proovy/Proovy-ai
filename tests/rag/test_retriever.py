from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from rag.retriever import search, set_vector_store
from rag.vector_store.base import BaseVectorStore
from rag.vector_store.mock import MockVectorStore


@pytest.fixture(autouse=True)
def _reset_vector_store() -> None:
    set_vector_store(MockVectorStore())


class _RecordingStore(BaseVectorStore):
    def __init__(self) -> None:
        self.last_top_k: Optional[int] = None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        return

    def search(
        self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        self.last_top_k = top_k
        return []


class _ErrorStore(BaseVectorStore):
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        return

    def search(
        self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        raise RuntimeError("boom")


class _ShapeStore(BaseVectorStore):
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        return

    def search(
        self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        return [{"content": "hello"}]


def test_search_empty_query_returns_empty() -> None:
    assert search("", context_texts=None) == []


def test_search_normalizes_top_k() -> None:
    store = _RecordingStore()
    set_vector_store(store)
    assert search("hi", top_k="-5") == []
    assert store.last_top_k == 0


def test_search_handles_store_errors() -> None:
    set_vector_store(_ErrorStore())
    assert search("hi") == []


def test_search_validates_doc_shape() -> None:
    set_vector_store(_ShapeStore())
    docs = search("hi")
    assert len(docs) == 1
    doc = docs[0]
    assert doc["text"] == "hello"
    assert doc["title"] == "Untitled"

