"""Memory layer stubs.

현재는 LangGraph용 실제 DB/스토어를 붙이지 않고, 서비스가 기동만 할 수 있도록
lifespan에서 사용하는 `initialize_database` / `initialize_store` 를 더미로 제공한다.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator


class InMemorySaver:
    """간단한 in-memory saver 더미.

    실제 LangGraph checkpointer 대신 자리표시자 역할만 한다.
    """

    async def setup(self) -> None:  # pragma: no cover - no-op
        return None


class InMemoryStore:
    """간단한 in-memory store 더미.

    실제 장기 저장소 대신 자리표시자 역할만 한다.
    """

    async def setup(self) -> None:  # pragma: no cover - no-op
        return None


@asynccontextmanager
async def initialize_database() -> AsyncIterator[InMemorySaver]:
    """단기 메모리(checkpointer) 초기화 더미.

    실제 DB 연결 대신 InMemorySaver 인스턴스를 넘겨준다.
    """

    saver = InMemorySaver()
    yield saver


@asynccontextmanager
async def initialize_store() -> AsyncIterator[InMemoryStore]:
    """장기 메모리(store) 초기화 더미.

    실제 DB 연결 대신 InMemoryStore 인스턴스를 넘겨준다.
    """

    store = InMemoryStore()
    yield store


__all__ = ["initialize_database", "initialize_store"]
