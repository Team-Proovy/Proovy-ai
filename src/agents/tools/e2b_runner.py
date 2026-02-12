from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from e2b_code_interpreter import Sandbox
from e2b_code_interpreter.models import Execution

from core.settings import settings

_SANDBOX: Sandbox | None = None
_SANDBOX_LOCK = None


def _get_sandbox(api_key: str, *, reuse: bool) -> Sandbox:
    global _SANDBOX, _SANDBOX_LOCK
    if not reuse:
        return Sandbox.create(api_key=api_key)
    if _SANDBOX_LOCK is None:
        import threading

        _SANDBOX_LOCK = threading.Lock()
    with _SANDBOX_LOCK:
        if _SANDBOX is None:
            _SANDBOX = Sandbox.create(api_key=api_key)
        return _SANDBOX


class E2BExecutionError(RuntimeError):
    """Raised when sandbox initialization or execution fails."""

    def __init__(self, message: str, execution: Optional[Execution] = None):
        super().__init__(message)
        self.execution = execution


@dataclass(slots=True)
class E2BExecutionResult:
    """Structured response returned after successful sandbox execution."""

    code: str
    text: Optional[str]
    stdout: list[str]
    stderr: list[str]
    execution: Execution


def _resolve_api_key() -> str:
    key: Optional[str] = None
    if settings.E2B_API_KEY:
        key = settings.E2B_API_KEY.get_secret_value()
    else:
        key = os.getenv("E2B_API_KEY") or os.getenv("E2B_KEY")

    if not key:
        raise E2BExecutionError("E2B API key is not configured in the environment.")

    return key.strip()


def run_python_with_e2b(
    code: str,
    *,
    envs: Optional[dict[str, str]] = None,
    timeout: Optional[float] = 60.0,
    request_timeout: Optional[float] = None,
    reuse_sandbox: bool = True,
) -> E2BExecutionResult:
    """Execute Python code inside an E2B sandbox and return stdout/stderr."""

    if not code.strip():
        raise ValueError("code must not be empty")

    api_key = _resolve_api_key()

    try:
        sandbox = _get_sandbox(api_key, reuse=reuse_sandbox)
        execution = sandbox.run_code(
            code,
            language="python",
            envs=envs,
            timeout=timeout,
            request_timeout=request_timeout,
        )
    except Exception as exc:
        raise E2BExecutionError("Failed to execute code inside E2B sandbox.") from exc

    if execution.error:
        error_message = f"{execution.error.name}: {execution.error.value}"
        raise E2BExecutionError(error_message, execution=execution)

    return E2BExecutionResult(
        code=code,
        text=execution.text,
        stdout=list(execution.logs.stdout),
        stderr=list(execution.logs.stderr),
        execution=execution,
    )


def get_user_friendly_error_message(error: E2BExecutionError | Exception) -> str:
    """
    E2B 에러를 사용자 친화적인 한국어 메시지로 변환합니다.

    Args:
        error: E2BExecutionError 또는 기타 Exception

    Returns:
        사용자에게 표시할 한국어 에러 메시지
    """
    error_str = str(error).lower()

    # 타임아웃 에러
    if "timeout" in error_str or "timed out" in error_str:
        return "코드 실행 시간이 초과되었습니다. 코드가 무한 루프에 빠졌거나 처리 시간이 너무 긴 것 같습니다."

    # API 키 설정 에러
    if "api key" in error_str:
        return "코드 실행 서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."

    # 문법 에러
    if "syntaxerror" in error_str:
        return "코드에 문법 오류가 있습니다. Python 문법을 다시 확인해주세요."

    # 이름 에러 (정의되지 않은 변수/함수)
    if "nameerror" in error_str:
        return "정의되지 않은 변수나 함수가 사용되었습니다. 변수명과 함수명을 확인해주세요."

    # 타입 에러
    if "typeerror" in error_str:
        return "잘못된 타입의 연산이 수행되었습니다. 데이터 타입을 확인해주세요."

    # 값 에러
    if "valueerror" in error_str:
        return "잘못된 값이 전달되었습니다. 입력값을 확인해주세요."

    # 인덱스 에러
    if "indexerror" in error_str:
        return "배열이나 리스트의 범위를 벗어났습니다. 인덱스를 확인해주세요."

    # 키 에러
    if "keyerror" in error_str:
        return "딕셔너리에 존재하지 않는 키를 참조했습니다. 키 이름을 확인해주세요."

    # 제로 나누기 에러
    if "zerodivisionerror" in error_str:
        return "0으로 나누기를 시도했습니다. 나누는 값을 확인해주세요."

    # 임포트 에러
    if "importerror" in error_str or "modulenotfounderror" in error_str:
        return "필요한 라이브러리를 불러올 수 없습니다. 사용 가능한 라이브러리인지 확인해주세요."

    # 속성 에러
    if "attributeerror" in error_str:
        return "객체에 존재하지 않는 속성이나 메서드를 사용했습니다."

    # 파일 에러
    if "filenotfounderror" in error_str:
        return "파일을 찾을 수 없습니다. 파일 경로를 확인해주세요."

    # 메모리 에러
    if "memoryerror" in error_str:
        return "메모리가 부족합니다. 데이터 크기를 줄여주세요."

    # 재귀 에러
    if "recursionerror" in error_str:
        return "재귀 호출이 너무 깊습니다. 재귀 조건을 확인해주세요."

    # 연결 에러
    if "connection" in error_str or "network" in error_str:
        return (
            "코드 실행 서버와의 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

    # E2B 샌드박스 실패
    if "sandbox" in error_str or "e2b" in error_str:
        return (
            "코드 실행 환경에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

    # 기본 메시지
    return "코드 실행 중 오류가 발생했습니다. 코드를 다시 확인해주세요."
