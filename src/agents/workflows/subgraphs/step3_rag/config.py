"""RAG subgraph configuration.

유사도 검색 관련 설정값 모음입니다.
환경변수로 오버라이드 가능하며 기본값과 단위는 아래에 명시되어 있습니다.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid int for %s: %r, using default %s", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s is negative (%s), using default %s", name, raw, default)
        return default
    return value


def _float_env(
    name: str, default: float, *, min_value: float | None = None, max_value: float | None = None
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid float for %s: %r, using default %s", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        logger.warning("%s below min (%s), using default %s", name, raw, default)
        return default
    if max_value is not None and value > max_value:
        logger.warning("%s above max (%s), using default %s", name, raw, default)
        return default
    return value


# TOP_K: 반환할 최대 문서 수 (정수)
TOP_K = _int_env("RAG_TOP_K", 3)

# MIN_DOCS: relevance fallback 최소 문서 수 (정수)
MIN_DOCS = _int_env("RAG_MIN_DOCS", 1)

# SCORE_THRESHOLD: score 임계값 (0.0 - 1.0 권장)
SCORE_THRESHOLD = _float_env(
    "RAG_SCORE_THRESHOLD", 0.75, min_value=0.0, max_value=1.0
)

# TRUNCATE_LIMIT: 문서/쿼리 자르기 한계 (문자 수)
TRUNCATE_LIMIT = _int_env("RAG_TRUNCATE_LIMIT", 280)

