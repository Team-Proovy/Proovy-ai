from __future__ import annotations

from typing import Any

import pytest

from rag.vector_store.pgvector import PgVectorStore


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[tuple[Any, Any]] = []
        self.executemany_calls: list[tuple[Any, Any]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: Any, params: Any = None) -> None:
        self.executed.append((query, params))

    def executemany(self, query: Any, params: Any = None) -> None:
        self.executemany_calls.append((query, params))

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class _FakeConnection:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._cursor = _FakeCursor(rows=rows)
        self.committed = False

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True


@pytest.fixture
def fake_connect(monkeypatch: pytest.MonkeyPatch) -> list[_FakeConnection]:
    conns: list[_FakeConnection] = []

    def _factory(*args: Any, **kwargs: Any) -> _FakeConnection:
        conn = _FakeConnection()
        conns.append(conn)
        return conn

    monkeypatch.setattr("rag.vector_store.pgvector.psycopg.connect", _factory)
    return conns


def test_pgvector_store_validates_constructor_args(fake_connect: list[_FakeConnection]) -> None:
    with pytest.raises(ValueError):
        PgVectorStore(dsn="", index_name="documents")
    with pytest.raises(ValueError):
        PgVectorStore(dsn="postgresql://u:p@localhost:5432/db", index_name="bad-name")
    with pytest.raises(ValueError):
        PgVectorStore(dsn="postgresql://u:p@localhost:5432/db", embedding_dim=0)


def test_pgvector_search_returns_normalized_docs(
    monkeypatch: pytest.MonkeyPatch, fake_connect: list[_FakeConnection]
) -> None:
    store = PgVectorStore(dsn="postgresql://u:p@localhost:5432/db")
    search_conn = _FakeConnection(
        rows=[
            {
                "id": "doc-1",
                "title": "t1",
                "text": "hello",
                "metadata": {"source": "unit"},
                "score": 0.91,
            }
        ]
    )
    monkeypatch.setattr(
        "rag.vector_store.pgvector.psycopg.connect",
        lambda *args, **kwargs: search_conn,
    )

    docs = store.search("hello", top_k=1, filters={"source": "unit"})
    assert len(docs) == 1
    assert docs[0]["id"] == "doc-1"
    assert docs[0]["text"] == "hello"
    assert docs[0]["metadata"]["source"] == "unit"
    assert isinstance(docs[0]["score"], float)


def test_pgvector_add_documents_upsert_called(
    monkeypatch: pytest.MonkeyPatch, fake_connect: list[_FakeConnection]
) -> None:
    store = PgVectorStore(dsn="postgresql://u:p@localhost:5432/db")
    write_conn = _FakeConnection()
    monkeypatch.setattr(
        "rag.vector_store.pgvector.psycopg.connect",
        lambda *args, **kwargs: write_conn,
    )

    store.add_documents(
        [
            {
                "id": "doc-a",
                "title": "A",
                "text": "alpha",
                "metadata": {"kind": "unit"},
                "embedding": [1.0, 0.0, 1.0],
            }
        ]
    )

    assert write_conn._cursor.executemany_calls
    assert write_conn.committed is True

