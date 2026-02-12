"""LangGraph PostgreSQL checkpointer integration.

PostgreSQL checkpointer를 사용하여 thread_id 기반으로 대화 상태를 저장/로드합니다.
Windows에서도 이벤트 루프 제약 없이 동작하도록 비동기 드라이버 대신
동기 PostgresSaver를 우선 사용합니다.

LangGraph의 AsyncPregelLoop 는 checkpointer 에서 비동기 메소드
``aget_tuple``, ``aput`` 등을 호출하지만, PostgresSaver 는
동기 메소드만 구현되어 있는 버전이 있어 NotImplementedError 가 발생할 수 있습니다.
이를 피하기 위해, 동기 saver 를 비동기 인터페이스로 감싸는 래퍼를 제공하여
async 메소드 호출을 내부적으로 스레드로 위임합니다.

PostgresSaver 초기화 실패 시 checkpointer 없이 AI 응답이 진행됩니다.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any, Optional
import logging

logger = logging.getLogger(__name__)


class InMemoryStore:
    """간단한 in-memory store 더미.

    실제 장기 저장소 대신 자리표시자 역할만 한다.
    """

    async def setup(self) -> None:  # pragma: no cover - no-op
        return None


class AsyncCheckpointerWrapper:
    """동기 checkpointer 를 LangGraph 가 기대하는 async 인터페이스로 감싸는 래퍼.

    LangGraph 의 AsyncPregelLoop 는 ``aget_tuple`` / ``aput`` 등의 비동기 메소드를
    호출한다. 일부 saver 구현(예: PostgresSaver, MemorySaver 의 특정 버전)은
    동기 메소드만 제공하므로, 여기서 비동기 메소드를 추가로 구현해준다.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 단순 위임
        return getattr(self._inner, name)

    # ---- sync 메소드 래핑 (필요 시 사용) ----
    def get_tuple(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - 단순 위임
        return getattr(self._inner, "get_tuple")(*args, **kwargs)

    def put(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - 단순 위임
        return getattr(self._inner, "put")(*args, **kwargs)

    def put_writes(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - 단순 위임
        """LangGraph 가 내부적으로 사용하는 sync put_writes 를 위임.

        PostgresSaver 는 put_writes 를 구현하고, base 클래스는 aput_writes 만
        NotImplemented 로 가지고 있을 수 있으므로, 여기서 명시적으로 위임한다.
        """

        return getattr(self._inner, "put_writes")(*args, **kwargs)

    # ---- async 메소드 구현 (LangGraph 가 실제로 사용하는 부분) ----
    async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
        """동기 get_tuple 을 백그라운드 스레드에서 실행.

        일부 saver 는 base 클래스 수준의 ``aget_tuple`` 를 가지고 있어
        NotImplementedError 를 발생시키므로, 존재 여부와 상관없이 항상
        동기 ``get_tuple`` 을 스레드에서 호출한다.
        """

        return await asyncio.to_thread(self._inner.get_tuple, *args, **kwargs)

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        """동기 put 을 백그라운드 스레드에서 실행.

        마찬가지로 base 클래스 수준의 ``aput`` 는 NotImplementedError 를
        던질 수 있으므로, 무조건 동기 ``put`` 을 스레드에서 호출한다.
        """

        return await asyncio.to_thread(self._inner.put, *args, **kwargs)

    async def aput_writes(self, *args: Any, **kwargs: Any) -> Any:
        """동기 put_writes 를 백그라운드 스레드에서 실행.

        LangGraph 의 checkpointer base 구현은 aput_writes 에서도
        NotImplementedError 를 던질 수 있으므로, 항상 동기 put_writes 를
        스레드에서 호출하도록 강제한다.
        """

        return await asyncio.to_thread(self._inner.put_writes, *args, **kwargs)


@asynccontextmanager
async def initialize_database() -> AsyncIterator[Optional[Any]]:
    """단기 메모리(checkpointer) 초기화.

    POSTGRES_URI가 설정되어 있으면 PostgresSaver(동기)를 사용합니다.
    실패 시 None을 반환하며, AI 응답은 checkpointer 없이 진행됩니다.
    """
    from core.settings import settings

    # POSTGRES_URI 가 설정되어 있으면 PostgresSaver 컨텍스트를 정상적으로 연다.
    # from_conn_string 는 contextmanager 를 반환하므로, "with" 로 들어가서
    # 실제 saver 인스턴스를 얻은 뒤 LangGraph checkpointer 로 넘겨야 한다.
    if settings.POSTGRES_URI:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import]

            logger.info(
                "Initializing PostgreSQL checkpointer with sync PostgresSaver..."
            )
            # from_conn_string 가 반환하는 contextmanager 안에서 실제 saver 를 획득
            with PostgresSaver.from_conn_string(settings.POSTGRES_URI) as saver:  # type: ignore[attr-defined]
                # 초기 실행 시 필요한 checkpoints 테이블 등이 없다면 생성한다.
                try:
                    saver.setup()  # type: ignore[attr-defined]
                    logger.info("PostgreSQL checkpointer schema setup completed")
                except Exception as setup_error:  # pragma: no cover - fallback path
                    logger.error(
                        "PostgreSQL checkpointer setup failed: %s",
                        setup_error,
                    )
                    raise

                logger.info("PostgreSQL checkpointer (sync) initialization completed")
                # AsyncPregelLoop 에서 필요한 async 메소드를 제공하도록 래퍼로 감싼다.
                async_saver = AsyncCheckpointerWrapper(saver)
                yield async_saver
                return
        except Exception as e:  # pragma: no cover - fallback path
            logger.error(
                "PostgresSaver initialization failed. Continuing without checkpointer: %s",
                e,
            )
            yield None
            return

    # POSTGRES_URI 가 설정되지 않은 경우
    logger.warning(
        "POSTGRES_URI not configured. Running without checkpointer (no conversation history persistence)."
    )
    yield None


@asynccontextmanager
async def initialize_store() -> AsyncIterator[InMemoryStore]:
    """장기 메모리(store) 초기화 더미.

    실제 DB 연결 대신 InMemoryStore 인스턴스를 넘겨준다.
    """

    store = InMemoryStore()
    yield store


__all__ = ["initialize_database", "initialize_store"]
