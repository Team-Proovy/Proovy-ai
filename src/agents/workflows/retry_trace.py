"""Retry trace utilities.

재시도(retry) 흐름에서 공통으로 사용하는 유틸리티 함수 모음.
- env_truthy : 환경 변수를 bool 값으로 읽는다.
- retry_trace_log : 재시도 관련 이벤트를 로깅한다.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def env_truthy(key: str, default: str = "false") -> bool:
    """환경 변수 *key* 의 값이 truthy 인지 반환한다.

    ``1 / true / yes / on`` (대소문자 무시)을 True 로 간주한다.
    환경 변수가 설정되어 있지 않으면 *default* 값을 기준으로 판단한다.
    """
    raw = (os.getenv(key, default) or "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def retry_trace_log(
    *,
    event: str,
    retry_count: int,
    node: str,
    state: Any,
    config: "RunnableConfig | None" = None,
    level: str = "info",
) -> None:
    """재시도 이벤트를 로깅한다.

    Args:
        event: 이벤트 식별자 (예: ``"before_retry_increment"``, ``"retry_limit_reached"``, ``"exit_retry_loop"``).
        retry_count: 현재 재시도 횟수.
        node: 이벤트가 발생한 LangGraph 노드 이름.
        state: 현재 AgentState.
        config: LangGraph RunnableConfig (선택).
        level: 로그 레벨 문자열 (``"info"`` 또는 ``"warning"``).
    """
    run_id = None
    if config and isinstance(config, dict):
        run_id = config.get("run_id") or (
            config.get("configurable") or {}
        ).get("run_id")

    msg = (
        f"[retry_trace] event={event!r} node={node!r} "
        f"retry_count={retry_count} run_id={run_id}"
    )

    log_fn = logger.warning if level == "warning" else logger.info
    log_fn(msg)
