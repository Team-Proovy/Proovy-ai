"""Solution feature subgraph.
해설 텍스트는 LLM으로 만들고, PDF는 e2b_runner에서 파이썬 코드로 생성한다.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from agents.state import AgentState, SolutionProgress, SolutionResult
from agents.tools import E2BExecutionError, run_python_with_e2b
from agents.workflows.utils import call_model, safe_json_loads
from schema.models import OpenRouterModelName

from .pdf_utils import (
    _build_pdf_code,
    _execute_pdf_locally,
    _extract_pdf_meta,
    _merge_local_font_env,
    _resolve_local_pdf_path,
)
from .problem_utils import (
    _align_explanations_to_problems,
    _build_pdf_entries,
    _collect_problems,
    _render_latex_to_plain,
    extract_problem_number,
)


CHUNK_SIZE_DEFAULT = 1
SOLUTION_JSON_EXAMPLE = '{"explanations":["해설1","해설2"],"chunk_summary":"요약"}'


def _ensure_progress(state: AgentState) -> SolutionProgress:
    progress = state.get("solution_progress")
    if isinstance(progress, dict):
        progress = SolutionProgress(**progress)
    if progress is None:
        progress = SolutionProgress(chunk_size=CHUNK_SIZE_DEFAULT)
    if not progress.chunk_size:
        progress.chunk_size = CHUNK_SIZE_DEFAULT
    state["solution_progress"] = progress
    return progress


def _ensure_solution_result(state: AgentState) -> SolutionResult:
    result = state.get("solution_result")
    if isinstance(result, dict):
        result = SolutionResult(**result)
    if result is None:
        result = SolutionResult()
    state["solution_result"] = result
    return result


def _record_pdf_success(
    solution_result: SolutionResult,
    tool_outputs: dict,
    *,
    pdf_path: str,
    pdf_name: str,
    pdf_size: Optional[int],
    pdf_base64: Optional[str],
    pdf_font: Optional[str],
    pdf_font_path: Optional[str],
    pdf_font_loaded: Optional[bool],
    stdout_lines: List[str],
    stderr_lines: List[str],
    source: str,
) -> None:
    solution_result.pdf_path = pdf_path
    solution_result.pdf_file_name = pdf_name
    solution_result.pdf_mime_type = "application/pdf"
    solution_result.pdf_file_size = pdf_size
    solution_result.pdf_error = None
    tool_outputs["solution_pdf"] = {
        "success": True,
        "source": source,
        "stdout": stdout_lines,
        "stderr": stderr_lines,
        "pdf_path": pdf_path,
        "file_name": pdf_name,
        "mime_type": "application/pdf",
        "file_size": pdf_size,
        "pdf_base64": pdf_base64,
        "pdf_font": pdf_font,
        "pdf_font_path": pdf_font_path,
        "pdf_font_loaded": pdf_font_loaded,
    }


def _record_pdf_failure(
    solution_result: SolutionResult,
    tool_outputs: dict,
    exc: Exception,
    stdout_lines: List[str],
    stderr_lines: List[str],
) -> None:
    solution_result.pdf_error = str(exc)
    tool_outputs["solution_pdf"] = {
        "success": False,
        "stdout": stdout_lines,
        "stderr": stderr_lines or [str(exc)],
    }


def _build_solution_prompts(problems: List[str]) -> Tuple[str, str]:
    system_prompt = (
        "You are a Korean tutor. Provide detailed explanations in Korean.\n"
        "Return ONLY valid JSON. No markdown, no extra text.\n"
        "Keys: explanations (list), chunk_summary (string).\n"
        "The length of explanations MUST equal the number of problems and keep order.\n"
        "Each explanation MUST start with the original problem number.\n"
        "Each explanation MUST include a first line formatted as '정답: ...' "
        "with the final answer."
    )
    user_prompt = (
        "다음 문제들에 대한 해설을 작성해 주세요.\n"
        "출력은 반드시 JSON만 반환하세요.\n\n"
        "예시 형식:\n"
        f"{SOLUTION_JSON_EXAMPLE}\n\n"
        "각 해설은 원본 문제 번호로 시작하고, 첫 줄에 '정답: 정답 내용 및 값'을 포함하세요.\n"
        "수식 내의 지수나 첨자를 주의 깊게 확인하고 원문의 선택지 내에서만 답을 고르세요.\n"
        "**중요: 만약 계산 결과가 주어진 선택지 ①~⑤ 중에 없다면, 자신의 계산 과정을 다시 검토하여 반드시 선택지 중 하나를 최종 정답으로 도출하세요. 절대로 선택지에 없는 값을 정답으로 쓰지 마세요.**\n"
        "정답은 보기/선택지 형식(예: ②, ㄱ·ㄴ·ㄷ, 14/81 등)을 그대로 쓰고, "
        "가능하면 결과를 한 번 검산해 주세요.\n\n"
        "문제 목록:\n"
        + json.dumps({"problems": problems}, ensure_ascii=False, indent=2)
    )
    return system_prompt, user_prompt


def _parse_solution_payload(raw: Any) -> Tuple[List[str], Optional[str]]:
    payload = safe_json_loads(raw) if isinstance(raw, str) else None
    if not isinstance(payload, dict):
        payload = {}
    explanations = payload.get("explanations") or []
    if not isinstance(explanations, list):
        explanations = [str(explanations)]
    explanations = [str(item).strip() for item in explanations if str(item).strip()]
    chunk_summary = str(payload.get("chunk_summary") or "").strip() or None
    return explanations, chunk_summary


def _request_explanations(
    problems: List[str],
    tool_outputs: dict,
) -> Tuple[List[str], Optional[str]]:
    system_prompt, user_prompt = _build_solution_prompts(problems)
    raw = call_model(OpenRouterModelName.GPT_5_MINI, system_prompt, user_prompt)
    explanations, chunk_summary = _parse_solution_payload(raw)

    if len(explanations) < len(problems):
        if isinstance(raw, str):
            tool_outputs["solution_llm_raw"] = raw
        retry_raw = call_model(OpenRouterModelName.GPT_5_MINI, system_prompt, user_prompt)
        if isinstance(retry_raw, str):
            tool_outputs["solution_llm_retry_raw"] = retry_raw
        retry_explanations, retry_summary = _parse_solution_payload(retry_raw)
        if len(retry_explanations) >= len(explanations):
            explanations = retry_explanations
            if retry_summary:
                chunk_summary = retry_summary

    if len(explanations) < len(problems):
        explanations.extend(["해설을 생성하지 못했습니다."] * (len(problems) - len(explanations)))

    return explanations, chunk_summary


def solution(state: AgentState) -> AgentState:
    """해설 기능 (청크 기반)."""
    print("---FEATURE: SOLUTION---")

    progress = _ensure_progress(state)
    solution_result = _ensure_solution_result(state)
    final_output = state.setdefault("final_output", {}) # [Fix] Early initialization
    
    problems = state.get("solution_chunks") or _collect_problems(state)
    state["solution_chunks"] = problems

    total_problems = len(problems)
    
    # 사용자가 '전부'를 원하면 전체 개수로, 아니면 기본값(1)으로 설정
    solve_all = state.get("solve_all", False)
    if solve_all:
        chunk_size = total_problems
    else:
        chunk_size = int(progress.chunk_size or CHUNK_SIZE_DEFAULT)
        
    total_chunks = math.ceil(total_problems / chunk_size) if total_problems and chunk_size else 0

    progress.total_problems = total_problems
    progress.total_chunks = total_chunks

    if total_chunks == 0:
        solution_result.guide = "해설할 문제가 없습니다."
        state["solution_result"] = solution_result
        state["prev_action"] = "Solution"
        return state
    if progress.done:
        solution_result.guide = solution_result.guide or "모든 청크가 이미 처리되었습니다."
        state["solution_result"] = solution_result
        state["prev_action"] = "Solution"
        return state

    current_chunk_index = max(0, min(int(progress.current_chunk or 0), total_chunks - 1))
    start = current_chunk_index * chunk_size
    end = min(start + chunk_size, total_problems)
    chunk_problems = problems[start:end]

    tool_outputs = state.setdefault("tool_outputs", {})
    explanations, chunk_summary = _request_explanations(chunk_problems, tool_outputs)
    explanations = _align_explanations_to_problems(chunk_problems, explanations)

    solution_result.guide = chunk_summary or solution_result.guide or "해설을 생성했습니다."
    
    # [개선] 현재 몇 번 문제를 풀었는지 가이드에 명시
    if chunk_problems:
        current_num = extract_problem_number(chunk_problems[0])
        if current_num:
            solution_result.guide = f"{current_num}번 문제의 해설을 완료했습니다. {solution_result.guide}"
            
    solution_result.chunk_index = current_chunk_index + 1
    solution_result.chunk_size = chunk_size
    solution_result.total_problems = total_problems
    solution_result.total_chunks = total_chunks
    solution_result.problems = chunk_problems
    solution_result.explanations = explanations
    solution_result.chunk_summary = chunk_summary

    # [개선] 안전한 전역 인덱스 업데이트 (역행 방지 및 메타데이터 기록)
    if chunk_problems:
        new_idx = end - 1
        old_idx = state.get("last_solved_index")
        if old_idx is None or new_idx > old_idx:
            state["last_solved_index"] = new_idx
            state["last_solved_index_ts"] = int(time.time() * 1000)
            state["last_solved_index_source"] = "Solution"

    pdf_file_name = f"solution_chunk_{solution_result.chunk_index}.pdf"
    emit_base64 = os.getenv("SOLUTION_EMIT_PDF_BASE64") == "1"
    
    render_latex_enabled = (
        os.getenv("SOLUTION_USE_MATH_RENDER", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    font_urls_env = os.getenv("SOLUTION_FONT_URLS")
    if font_urls_env:
        font_urls = [item.strip() for item in font_urls_env.split(",") if item.strip()]
    else:
        font_url = os.getenv("SOLUTION_FONT_URL")
        font_urls = (
            [font_url]
            if font_url
            else [
                "https://cdn.jsdelivr.net/gh/google/fonts/ofl/notosanskr/NotoSansKR-Regular.ttf",
                "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR-Regular.ttf",
            ]
        )
    font_base64 = os.getenv("SOLUTION_FONT_BASE64")
    if not font_base64:
        font_base64_path = os.getenv("SOLUTION_FONT_BASE64_PATH")
        if font_base64_path:
            try:
                with open(font_base64_path, "r", encoding="utf-8") as handle:
                    font_base64 = re.sub(r"\s+", "", handle.read())
            except OSError:
                font_base64 = None

    pdf_chunk_problems = chunk_problems
    pdf_explanations = explanations

    if not render_latex_enabled:
        pdf_chunk_problems = [_render_latex_to_plain(p) for p in chunk_problems]
        pdf_explanations = [_render_latex_to_plain(e) for e in explanations]

    pdf_payload = {
        "title": f"Solution Chunk {solution_result.chunk_index}/{total_chunks}",
        "entries": _build_pdf_entries(pdf_chunk_problems, pdf_explanations),
        "summary": None,
        "pdf_path": f"/home/user/{pdf_file_name}",
        "file_name": pdf_file_name,
        "emit_base64": emit_base64,
        "font_urls": font_urls,
        "font_base64": font_base64,
    }
    pdf_code = _build_pdf_code(pdf_payload)
    sandbox_envs: dict[str, str] = {}
    font_path_env = os.getenv("SOLUTION_FONT_PATH")
    if font_path_env:
        sandbox_envs["SOLUTION_FONT_PATH"] = font_path_env
    install_deps_env = os.getenv("SOLUTION_E2B_INSTALL_DEPS")
    if install_deps_env:
        sandbox_envs["SOLUTION_E2B_INSTALL_DEPS"] = install_deps_env
    
    use_math_render_env = os.getenv("SOLUTION_USE_MATH_RENDER")
    if use_math_render_env:
        sandbox_envs["SOLUTION_USE_MATH_RENDER"] = use_math_render_env

    install_deps = os.getenv("SOLUTION_E2B_INSTALL_DEPS", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }
    timeout_env = os.getenv("SOLUTION_E2B_TIMEOUT")
    timeout = None
    if timeout_env:
        try:
            timeout = float(timeout_env)
        except ValueError:
            timeout = None
    if timeout is None:
        timeout = 300.0 if install_deps else 180.0
    request_timeout = timeout + 60.0 if timeout else None
    reuse_env = os.getenv("SOLUTION_E2B_REUSE", "").strip().lower()
    reuse_sandbox = reuse_env not in {"0", "false", "no", "off"}
    
    attempts: List[Tuple[dict[str, str], bool]] = [(sandbox_envs, reuse_sandbox)]
    if reuse_sandbox:
        attempts.append((sandbox_envs, False))
    if render_latex_enabled:
        fallback_envs = dict(sandbox_envs)
        fallback_envs["SOLUTION_USE_MATH_RENDER"] = "0"
        attempts.append((fallback_envs, False))
    execution = None
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    pdf_error: Exception | None = None
    pdf_success = False
    try:
        execution = None
        last_exc: Exception | None = None
        for envs, reuse_flag in attempts:
            try:
                execution = run_python_with_e2b(
                    pdf_code,
                    envs=envs or None,
                    timeout=timeout,
                    request_timeout=request_timeout,
                    reuse_sandbox=reuse_flag,
                )
                last_exc = None
                break
            except E2BExecutionError as exc:
                last_exc = exc
                if "UnexpectedEndOfExecution" in str(exc):
                    continue
                raise
        if execution is None and last_exc is not None:
            raise last_exc

        raw_stdout = execution.stdout if hasattr(execution, "stdout") else execution
        if isinstance(raw_stdout, str):
            stdout_lines = raw_stdout.splitlines()
        elif isinstance(raw_stdout, list):
            stdout_lines = raw_stdout
        else:
            stdout_lines = [str(raw_stdout)]

        raw_stderr = execution.stderr if hasattr(execution, "stderr") else None
        if isinstance(raw_stderr, str):
            stderr_lines = raw_stderr.splitlines()
        elif isinstance(raw_stderr, list):
            stderr_lines = raw_stderr
        else:
            stderr_lines = [] if raw_stderr is None else [str(raw_stderr)]

        (
            pdf_path,
            pdf_name,
            pdf_size,
            pdf_base64,
            pdf_font,
            pdf_font_path,
            pdf_font_loaded,
        ) = _extract_pdf_meta(stdout_lines)
        pdf_path = pdf_path or pdf_payload["pdf_path"]
        pdf_name = pdf_name or pdf_file_name
        _record_pdf_success(
            solution_result,
            tool_outputs,
            pdf_path=pdf_path,
            pdf_name=pdf_name,
            pdf_size=pdf_size,
            pdf_base64=pdf_base64,
            pdf_font=pdf_font,
            pdf_font_path=pdf_font_path,
            pdf_font_loaded=pdf_font_loaded,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            source="e2b",
        )
        pdf_success = True
    except Exception as exc:
        pdf_error = exc
        stdout_lines = []
        stderr_lines = []
        try:
            execution_obj = None
            if isinstance(exc, E2BExecutionError) and getattr(exc, "execution", None):
                execution_obj = exc.execution
            elif "execution" in locals() and execution is not None:
                execution_obj = execution

            if execution_obj is not None:
                raw_stdout = getattr(execution_obj, "stdout", None)
                raw_stderr = getattr(execution_obj, "stderr", None)
                if raw_stdout is None and hasattr(execution_obj, "logs"):
                    raw_stdout = getattr(execution_obj.logs, "stdout", None)
                if raw_stderr is None and hasattr(execution_obj, "logs"):
                    raw_stderr = getattr(execution_obj.logs, "stderr", None)

                if isinstance(raw_stdout, str):
                    stdout_lines = raw_stdout.splitlines()
                elif isinstance(raw_stdout, list):
                    stdout_lines = raw_stdout

                if isinstance(raw_stderr, str):
                    stderr_lines = raw_stderr.splitlines()
                elif isinstance(raw_stderr, list):
                    stderr_lines = raw_stderr
        except Exception:
            pass

    if not pdf_success:
        local_fallback_enabled = (
            os.getenv("SOLUTION_LOCAL_FALLBACK", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        if local_fallback_enabled:
            local_payload = dict(pdf_payload)
            local_payload["pdf_path"] = _resolve_local_pdf_path(pdf_file_name)
            local_code = _build_pdf_code(local_payload)
            local_exc: Exception | None = None
            for envs, _reuse_flag in attempts:
                local_envs = _merge_local_font_env(
                    envs,
                    pdf_path=local_payload["pdf_path"],
                )
                ok, local_stdout, local_stderr, local_exc = _execute_pdf_locally(
                    local_code,
                    local_envs,
                )
                if ok:
                    stdout_lines = local_stdout
                    stderr_lines = local_stderr
                    (
                        pdf_path,
                        pdf_name,
                        pdf_size,
                        pdf_base64,
                        pdf_font,
                        pdf_font_path,
                        pdf_font_loaded,
                    ) = _extract_pdf_meta(stdout_lines)
                    pdf_path = pdf_path or local_payload["pdf_path"]
                    pdf_name = pdf_name or pdf_file_name
                    _record_pdf_success(
                        solution_result,
                        tool_outputs,
                        pdf_path=pdf_path,
                        pdf_name=pdf_name,
                        pdf_size=pdf_size,
                        pdf_base64=pdf_base64,
                        pdf_font=pdf_font,
                        pdf_font_path=pdf_font_path,
                        pdf_font_loaded=pdf_font_loaded,
                        stdout_lines=stdout_lines,
                        stderr_lines=stderr_lines,
                        source="local",
                    )
                    pdf_success = True
                    pdf_error = None
                    solution_result.pdf_error = None
                    
                    final_output["final_answer"] = (
                        f"요청하신 해설지 PDF 생성을 완료했습니다.\n\n"
                        f"파일 정보\n"
                        f"- 파일명: {pdf_name}\n"
                        f"- 저장 경로: {pdf_path}\n"
                        f"- 파일 크기: {pdf_size or 0} bytes\n\n"
                        f"내용 요약: {solution_result.chunk_summary or '해설 생성이 완료되었습니다.'}"
                    )
                    break
                if local_stdout:
                    stdout_lines = local_stdout
                if local_stderr:
                    stderr_lines = local_stderr
            if not pdf_success and local_exc is not None:
                if pdf_error is not None and str(pdf_error) != str(local_exc):
                    pdf_error = RuntimeError(
                        f"{pdf_error} | local_fallback: {local_exc}"
                    )
                else:
                    pdf_error = local_exc

    if not pdf_success:
        if pdf_error is None:
            pdf_error = RuntimeError("PDF generation failed.")
        _record_pdf_failure(
            solution_result, tool_outputs, pdf_error, stdout_lines, stderr_lines
        )

    progress.current_chunk = current_chunk_index + 1
    progress.done = progress.current_chunk >= total_chunks
    remaining_count = max(0, total_problems - (progress.current_chunk * chunk_size))
    state["solution_progress"] = progress

    final_output["solution"] = {
        "chunk_index": solution_result.chunk_index,
        "chunk_size": solution_result.chunk_size,
        "total_problems": solution_result.total_problems,
        "total_chunks": solution_result.total_chunks,
        "problems": [_render_latex_to_plain(p) for p in solution_result.problems],
        "explanations": [_render_latex_to_plain(e) for e in solution_result.explanations],
        "chunk_summary": solution_result.chunk_summary,
        "pdf_path": solution_result.pdf_path,
        "pdf_file_name": solution_result.pdf_file_name,
        "pdf_mime_type": solution_result.pdf_mime_type,
        "pdf_file_size": solution_result.pdf_file_size,
        "pdf_error": solution_result.pdf_error,
        "next_chunk_available": not progress.done,
        "remaining_count": remaining_count,
    }
    if tool_outputs.get("solution_pdf"):
        final_output["solution"]["pdf_font"] = tool_outputs["solution_pdf"].get("pdf_font")
        final_output["solution"]["pdf_font_path"] = tool_outputs["solution_pdf"].get("pdf_font_path")
        final_output["solution"]["pdf_font_loaded"] = tool_outputs["solution_pdf"].get("pdf_font_loaded")
    if emit_base64:
        final_output["solution"]["pdf_base64"] = tool_outputs.get("solution_pdf", {}).get("pdf_base64")
    if not progress.done:
        remaining_problems = problems[end:]
        next_problem_text = remaining_problems[0] if remaining_problems else None
        next_num = extract_problem_number(next_problem_text) if next_problem_text else None
        
        suggestion_text = f"다음 {next_num}번 문제도 풀어드릴까요?" if next_num else f"다음 {len(remaining_problems)}문제도 풀어드릴까요?"
        if not next_num and len(remaining_problems) == 1:
            suggestion_text = "다음 문제도 풀어드릴까요?"

        final_output["solution"]["suggestions"] = [
            suggestion_text,
            "전체 요약본이 필요하신가요?",
        ]
        if len(remaining_problems) > 1:
            final_output["solution"]["suggestions"].insert(1, f"남은 {len(remaining_problems)}문제 모두 풀기")
            
    state["final_output"] = final_output
    state["solution_result"] = solution_result
    state["prev_action"] = "Solution"
    return state


builder = StateGraph(AgentState)
builder.add_node("Solution", solution)
builder.set_entry_point("Solution")
builder.add_edge("Solution", END)

graph = builder.compile()
