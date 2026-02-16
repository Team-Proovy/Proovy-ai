"""Solve feature subgraph."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import List

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from agents.prompts.solve_prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    FINAL_SUMMARY_SYSTEM_PROMPT,
    STRATEGY_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    build_analysis_user_prompt,
    build_strategy_user_prompt,
    build_writer_user_prompt,
)
from agents.state import (
    AgentState,
    ComputationSummary,
    ProblemAnalysis,
    SolveResult,
    SolveStrategy,
)
from agents.tools import (
    E2BExecutionError,
    get_user_friendly_error_message,
    run_python_with_e2b,
)
from agents.workflows.utils import (
    call_model,
    call_model_by_difficulty,
    classify_difficulty,
    ensure_str_list,
    extract_ocr_text,
    get_conversation_summary,
    get_difficulty_from_state,
    recent_user_context,
    safe_json_loads,
    set_difficulty_in_state,
)
from schema.models import OpenRouterModelName


def _ensure_solve_result(state: AgentState) -> SolveResult:
    solve_result = state.get("solve_result")
    if isinstance(solve_result, dict):
        solve_result = SolveResult(**solve_result)
    if solve_result is None:
        solve_result = SolveResult()
    state["solve_result"] = solve_result
    return solve_result


def _env_truthy(key: str, default: str = "0") -> bool:
    value = os.getenv(key, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _run_python_locally(
    code: str,
    *,
    timeout: float = 30.0,
) -> tuple[bool, List[str], List[str], str | None, str | None]:
    """Fallback local execution for generated Python code."""

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(code)
            temp_file = handle.name

        runtime_env: dict[str, str] = {"PYTHONIOENCODING": "utf-8"}
        allowed_env_keys = {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "HOME",
            "USERPROFILE",
            "TMP",
            "TEMP",
            "PYTHONHOME",
        }
        for key in allowed_env_keys:
            value = os.environ.get(key)
            if value is not None:
                runtime_env[key] = value

        result = subprocess.run(
            [sys.executable, temp_file],
            env=runtime_env,
            capture_output=True,
            text=False,
            timeout=timeout,
        )
        stdout = result.stdout.decode("utf-8", errors="ignore").splitlines()
        stderr = result.stderr.decode("utf-8", errors="ignore").splitlines()
        text_output = "\n".join(stdout).strip() or None
        return result.returncode == 0, stdout, stderr, text_output, None
    except Exception as exc:  # pragma: no cover - defensive fallback
        return False, [], [], None, str(exc)
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


def analyze_problem(state: AgentState) -> AgentState:
    print("---FEATURE: SOLVE / ANALYSIS---")
    solve_result = _ensure_solve_result(state)
    user_text = recent_user_context(state, max_messages=3, include_assistant=True)
    ocr_text = extract_ocr_text(state)

    indexed_problem_text = ""
    indexed_problem_number = None
    problems = state.get("problems") or []
    if isinstance(problems, list):
        idx = int(state.get("current_problem_index", 0) or 0)
        if 0 <= idx < len(problems):
            item = problems[idx]
            if isinstance(item, dict):
                indexed_problem_text = str(item.get("text") or "").strip()
                indexed_problem_number = item.get("number")
            else:
                indexed_problem_text = str(item).strip()
    # 난이도 분류 (Solve 단계에서 직접 수행)
    combined_question = user_text or ""
    if ocr_text:
        combined_question = f"{combined_question}\n{ocr_text}".strip()
    difficulty = classify_difficulty(combined_question)
    set_difficulty_in_state(state, difficulty)
    print(f"---SOLVE: DIFFICULTY CLASSIFICATION RESULT {difficulty}---")

    # 이전 대화 맥락 수집 (멀티턴 지원)
    conversation_context = get_conversation_summary(state, max_chars=1000)

    # 대화 맥락이 있으면 포함
    context_section = ""
    if conversation_context:
        context_section = (
            "Previous conversation context (for reference if user mentions previous "
            f"problems):\n{conversation_context}"
        )

    analysis_prompt = build_analysis_user_prompt(
        user_text=user_text or "N/A",
        ocr_text=ocr_text or "N/A",
        context_section=context_section,
        indexed_problem_number=(
            str(indexed_problem_number)
            if indexed_problem_number is not None
            else "N/A"
        ),
        indexed_problem_text=indexed_problem_text or "N/A",
    )

    raw_response = call_model(
        OpenRouterModelName.GPT_5_MINI,
        ANALYSIS_SYSTEM_PROMPT,
        analysis_prompt,
    )
    payload = safe_json_loads(raw_response)
    problem_statement = (
        payload.get("problem")
        or indexed_problem_text
        or user_text
        or ocr_text
        or "문제가 명확하지 않습니다."
    )
    analysis = ProblemAnalysis(
        problem_statement=str(problem_statement).strip(),
        domain=str(payload.get("domain") or "general").strip(),
        knowns=ensure_str_list(payload.get("knowns")),
        unknowns=ensure_str_list(payload.get("unknowns")),
        laws=ensure_str_list(payload.get("laws")),
        constraints=ensure_str_list(payload.get("constraints")),
        hints=ensure_str_list(payload.get("hints")),
    )
    solve_result.analysis = analysis
    solve_result.problem = analysis.problem_statement

    state["solve_result"] = solve_result
    state["prev_action"] = "Solve_Analysis"
    return state


def plan_solution_strategy(state: AgentState) -> AgentState:
    print("---FEATURE: SOLVE / STRATEGY---")
    solve_result = _ensure_solve_result(state)
    if not solve_result.analysis:
        raise ValueError("Problem analysis missing; cannot plan strategy.")

    analysis_payload = json.dumps(
        solve_result.analysis.model_dump(),
        ensure_ascii=False,
        indent=2,
    )
    strategy_prompt = build_strategy_user_prompt(analysis_payload)

    raw_response = call_model(
        OpenRouterModelName.GPT_5_1_CODEX_MINI,
        STRATEGY_SYSTEM_PROMPT,
        strategy_prompt,
    )
    payload = safe_json_loads(raw_response)
    generated_code = str(
        payload.get("generated_code") or payload.get("code") or ""
    ).strip()
    if not generated_code:
        raise ValueError("전략 노드가 실행 코드를 생성하지 못했습니다.")

    strategy = SolveStrategy(
        summary=str(
            payload.get("summary") or payload.get("plan") or "계획 요약 없음"
        ).strip(),
        steps=ensure_str_list(payload.get("steps")),
        generated_code=generated_code,
    )
    solve_result.strategy = strategy
    solve_result.steps = strategy.steps

    tool_outputs = state.setdefault("tool_outputs", {})
    tool_outputs["solve_generated_code"] = generated_code
    state["tool_outputs"] = tool_outputs

    state["solve_result"] = solve_result
    state["prev_action"] = "Solve_Strategy"
    state["next_action"] = "Solve_Computation"
    return state


def execute_strategy(state: AgentState) -> AgentState:
    print("---FEATURE: SOLVE / COMPUTATION---")
    solve_result = _ensure_solve_result(state)
    if not solve_result.strategy:
        raise ValueError("전략 정보가 없어 계산을 실행할 수 없습니다.")

    code = solve_result.strategy.generated_code
    tool_outputs = state.setdefault("tool_outputs", {})
    tool_outputs["solve_execution_status"] = "running"
    tool_outputs["solve_execution_backend"] = "e2b"
    state["tool_outputs"] = tool_outputs

    success = True
    stdout: List[str]
    stderr: List[str]
    text_output: str | None
    execution_backend = "e2b"

    try:
        execution = run_python_with_e2b(code)
        stdout = execution.stdout
        stderr = execution.stderr
        text_output = execution.text
    except E2BExecutionError as exc:
        success = False
        stdout = []
        stderr = [get_user_friendly_error_message(exc)]
        text_output = None
        if exc.execution:
            text_output = getattr(exc.execution, "text", None) or text_output
            if getattr(exc.execution, "logs", None):
                stdout.extend(getattr(exc.execution.logs, "stdout", []))
                stderr.extend(getattr(exc.execution.logs, "stderr", []))
        else:
            # Security default: local execution of model-generated code is opt-in only.
            allow_local_fallback = _env_truthy("SOLVE_LOCAL_PYTHON_FALLBACK", "0")
            if allow_local_fallback:
                execution_backend = "local"
                local_ok, local_stdout, local_stderr, local_text, local_error = (
                    _run_python_locally(code)
                )
                success = local_ok
                stdout = local_stdout
                stderr = list(local_stderr)
                if local_error:
                    stderr.append(f"Local execution error: {local_error}")
                stderr.append(f"E2B fallback reason: {str(exc)}")
                text_output = local_text
            else:
                stderr.append(f"Execution error: {str(exc)}")
    
    execution_summary = ComputationSummary(
        success=success,
        stdout=stdout,
        stderr=stderr,
        text=text_output,
    )
    solve_result.computation = execution_summary
    tool_outputs["solve_execution"] = execution_summary.model_dump()
    tool_outputs["solve_execution_backend"] = execution_backend
    tool_outputs["solve_execution_status"] = "success" if success else "failed"
    state["tool_outputs"] = tool_outputs

    analysis_dump = solve_result.analysis.model_dump() if solve_result.analysis else {}
    strategy_dump = solve_result.strategy.model_dump()
    computation_dump = execution_summary.model_dump()
    final_prompt = json.dumps(
        {
            "analysis": analysis_dump,
            "strategy": strategy_dump,
            "computation": computation_dump,
        },
        ensure_ascii=False,
        indent=2,
    )
    # 난이도 기반 모델로 결과 요약 (복잡한 문제는 더 강력한 모델 사용)
    difficulty = get_difficulty_from_state(state)
    print(f"→ Summarizing with difficulty: {difficulty}")
    summary_raw = call_model_by_difficulty(
        state,
        FINAL_SUMMARY_SYSTEM_PROMPT,
        final_prompt,
    )
    summary_payload = safe_json_loads(summary_raw)
    solve_result.answer = str(
        summary_payload.get("answer")
        or summary_payload.get("result")
        or "답을 정리할 수 없습니다."
    ).strip()
    steps = ensure_str_list(summary_payload.get("steps"))
    solve_result.steps = steps or solve_result.steps
    latex_value = str(summary_payload.get("latex") or "").strip()
    solve_result.latex = latex_value or solve_result.latex
    final_summary = str(
        summary_payload.get("summary")
        or summary_payload.get("explanation")
        or solve_result.answer
    ).strip()

    final_output = state.setdefault("final_output", {})
    final_output["solve"] = {
        "analysis": analysis_dump,
        "strategy": strategy_dump,
        "computation": computation_dump,
        "execution_status": tool_outputs.get("solve_execution_status"),
        "execution_backend": tool_outputs.get("solve_execution_backend"),
        "answer": solve_result.answer,
        "steps": solve_result.steps,
        "latex": solve_result.latex,
        "summary": final_summary,
    }
    state["final_output"] = final_output

    state["solve_result"] = solve_result
    state["prev_action"] = "Solve_Computation"
    return state


def solve_writer(state: AgentState) -> AgentState:
    """Solve 결과를 즉시 사용자 친화적인 한국어 텍스트로 변환하는 Writer 노드.

    이미 생성된 구조화 결과(answer, steps, latex)를 경량 LLM으로 포맷팅하여
    토큰 스트리밍이 가능하도록 합니다. (LangGraph가 자동 감지)
    """
    print("---FEATURE: SOLVE / WRITER---")
    solve_result = _ensure_solve_result(state)

    # 기존 결과를 JSON으로 직렬화
    retry_count = state.get("retry_count", 0) or 0

    # LLM에게 주어지는 구조화 데이터
    solve_data = {
        "answer": solve_result.answer or "답이 없습니다",
        "steps": solve_result.steps or [],
        "latex": solve_result.latex or "",
        "retry_count": retry_count,
        "computation_success": solve_result.computation.success
        if solve_result.computation
        else True,
        "computation_errors": solve_result.computation.stderr[:3]
        if solve_result.computation and not solve_result.computation.success
        else [],
    }

    serialized_data = json.dumps(solve_data, ensure_ascii=False, indent=2)

    # 경량 LLM으로 포맷팅 (토큰 스트리밍 가능)
    user_prompt = build_writer_user_prompt(serialized_data)

    # LLM 호출 (토큰 스트리밍 가능, tags=[] 명시)
    formatted_content = call_model(
        OpenRouterModelName.GPT_5_MINI,
        WRITER_SYSTEM_PROMPT,
        user_prompt,
        tags=[],  # 스트리밍 허용
    ).strip()

    # partial_responses에 누적 (FinalResponse가 활용)
    partial_responses = state.get("partial_responses", [])
    if partial_responses is None:
        partial_responses = []

    partial_responses.append(
        {
            "feature": "Solve",
            "content": formatted_content,
            "has_answer": bool(solve_result.answer),
            "retry_count": retry_count,
        }
    )
    state["partial_responses"] = partial_responses

    # AIMessage를 messages에 추가하여 즉시 스트리밍 및 상태 저장
    if formatted_content:
        ai_msg = AIMessage(content=formatted_content)
        state["messages"] = (state.get("messages") or []) + [ai_msg]

    state["prev_action"] = "Solve_Writer"

    return state


builder = StateGraph(AgentState)
# 중간 분석/전략/계산 노드는 스트리밍 차단 (skip_stream)
builder.add_node("Solve_Analysis", analyze_problem, tags=["skip_stream"])
builder.add_node("Solve_Strategy", plan_solution_strategy, tags=["skip_stream"])
builder.add_node("Solve_Computation", execute_strategy, tags=["skip_stream"])
# Writer 노드는 스트리밍 허용 (태그 없음)
builder.add_node("Solve_Writer", solve_writer)

builder.set_entry_point("Solve_Analysis")

# 항상 코드 실행: Analysis → Strategy → Computation → Writer → END
builder.add_edge("Solve_Analysis", "Solve_Strategy")
builder.add_edge("Solve_Strategy", "Solve_Computation")
builder.add_edge("Solve_Computation", "Solve_Writer")
builder.add_edge("Solve_Writer", END)

graph = builder.compile()
