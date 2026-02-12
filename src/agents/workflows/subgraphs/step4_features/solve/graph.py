"""Solve feature subgraph."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import List

from langgraph.graph import END, StateGraph

from agents.state import (
    AgentState,
    ComputationSummary,
    ProblemAnalysis,
    SolveResult,
    SolveStrategy,
)
from agents.tools import E2BExecutionError, run_python_with_e2b
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

        result = subprocess.run(
            [sys.executable, temp_file],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
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
        context_section = f"\n\nPrevious conversation context (for reference if user mentions previous problems):\n{conversation_context}\n"

    analysis_prompt = f"""
User input (may be Korean or English):
{user_text or "N/A"}

OCR extracted text (if any):
{ocr_text or "N/A"}
{context_section}
Task: Analyze only the first explicit STEM problem you can find and respond in English.
If user refers to a previous problem (e.g., '이전 문제', '방금 푼 문제'), use the conversation context to identify it.
""".strip()

    system_prompt = (
        "You are a STEM problem analyst. Extract only the first explicit problem. "
        "If the user refers to a previous problem, find it from the conversation context. "
        "Return structured JSON with keys: problem, domain, knowns, unknowns, laws, constraints, hints. "
        "Always produce arrays for multi-valued fields and keep all text in concise English."
    )

    raw_response = call_model(
        OpenRouterModelName.GPT_5_MINI,
        system_prompt,
        analysis_prompt,
    )
    payload = safe_json_loads(raw_response)
    problem_statement = (
        payload.get("problem") or user_text or ocr_text or "문제가 명확하지 않습니다."
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
    strategy_prompt = f"""
Here is the problem analysis (in English):
{analysis_payload}

Task: Plan a step-by-step solution strategy in English and generate Python code that follows the plan.
The code must NOT use external network access or file I/O; focus only on numeric computation and symbolic manipulation.
""".strip()

    system_prompt = (
        "You are a meticulous STEM strategist. Produce JSON with keys summary, steps, generated_code. "
        "Write summary and steps in clear English. 'steps' must be an ordered list guiding the solution, "
        "and 'generated_code' must be runnable Python that follows those steps."
    )

    raw_response = call_model(
        OpenRouterModelName.GPT_5_1_CODEX_MINI,
        system_prompt,
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
        allow_local_fallback = _env_truthy("SOLVE_LOCAL_PYTHON_FALLBACK", "1")
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
            success = False
            stdout = []
            stderr = [str(exc)]
            text_output = None
            if exc.execution:
                text_output = getattr(exc.execution, "text", None) or text_output
                if getattr(exc.execution, "logs", None):
                    stdout.extend(getattr(exc.execution.logs, "stdout", []))
                    stderr.extend(getattr(exc.execution.logs, "stderr", []))

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
    system_prompt = (
        "You are a STEM tutor who must provide the final solution in Korean. "
        "Return JSON with keys answer (concise result), steps (2-5 bullet reminders), "
        "latex (optional final expression), and summary (one short explanation)."
    )

    # 난이도 기반 모델로 결과 요약 (복잡한 문제는 더 강력한 모델 사용)
    difficulty = get_difficulty_from_state(state)
    print(f"→ Summarizing with difficulty: {difficulty}")
    summary_raw = call_model_by_difficulty(
        state,
        system_prompt,
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
    system_prompt = (
        "너는 수학 문제 풀이 결과를 한국어로 정리하는 전문가야. "
        "아래 JSON 데이터를 보고 사용자가 읽기 좋게 Markdown 형식으로 변환해 줘. "
        "다음 규칙을 따라:\n"
        "1. retry_count > 0이면 '다시 계산해본 결과입니다' 문구 추가\n"
        "2. answer는 '**답:** {answer}' 형식\n"
        "3. steps가 있으면 '**풀이 과정:**' + 번호 리스트\n"
        "4. latex가 있으르면 '**수식:** ${latex}$' 형식\n"
        "5. computation_success가 false면 '⚠️ 계산 중 오류...' + 에러 3줄\n"
        "불필요한 설명 없이 간결하게 작성하고, 주어진 데이터만 사용해."
    )

    user_prompt = f"다음 데이터를 포맷팅해 주세요:\n\n{serialized_data}"

    # LLM 호출 (토큰 스트리밍 가능, tags=[] 명시)
    formatted_content = call_model(
        OpenRouterModelName.GPT_5_MINI,
        system_prompt,
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

    # AIMessage는 call_model이 자동으로 생성해주므로 직접 추가 불필요
    # (call_model 내부에서 LLM 호출 시 LangGraph가 AIMessage 자동 추가)

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
