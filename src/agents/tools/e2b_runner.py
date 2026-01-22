from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from e2b_code_interpreter import Sandbox
from e2b_code_interpreter.models import Execution

from core.settings import settings


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
) -> E2BExecutionResult:
    """Execute Python code inside an E2B sandbox and return stdout/stderr."""

    if not code.strip():
        raise ValueError("code must not be empty")

    api_key = _resolve_api_key()

    try:
        with Sandbox.create(api_key=api_key) as sandbox:
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
