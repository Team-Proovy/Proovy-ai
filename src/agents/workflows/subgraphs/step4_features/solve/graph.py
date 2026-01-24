"""Solve feature subgraph."""

from __future__ import annotations

import json
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
    ensure_str_list,
    extract_ocr_text,
    recent_user_context,
    safe_json_loads,
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


def analyze_problem(state: AgentState) -> AgentState:
    print("---FEATURE: SOLVE / ANALYSIS---")
    solve_result = _ensure_solve_result(state)
    user_text = recent_user_context(state)
    ocr_text = extract_ocr_text(state)
    analysis_prompt = f"""
User input (may be Korean or English):
{user_text or "N/A"}

OCR extracted text (if any):
{ocr_text or "N/A"}

Task: Analyze only the first explicit STEM problem you can find and respond in English.
""".strip()

    system_prompt = (
        "You are a STEM problem analyst. Extract only the first explicit problem. "
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
    state["next_action"] = "Solve_Strategy"
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
    success = True
    stdout: List[str]
    stderr: List[str]
    text_output: str | None

    try:
        execution = run_python_with_e2b(code)
        stdout = execution.stdout
        stderr = execution.stderr
        text_output = execution.text
    except E2BExecutionError as exc:
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

    summary_raw = call_model(
        OpenRouterModelName.GPT_5_MINI,
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
        "answer": solve_result.answer,
        "steps": solve_result.steps,
        "latex": solve_result.latex,
        "summary": final_summary,
    }
    state["final_output"] = final_output

    state["solve_result"] = solve_result
    state["prev_action"] = "Solve_Computation"
    return state


builder = StateGraph(AgentState)
builder.add_node("Solve_Analysis", analyze_problem)
builder.add_node("Solve_Strategy", plan_solution_strategy)
builder.add_node("Solve_Computation", execute_strategy)
builder.set_entry_point("Solve_Analysis")
builder.add_edge("Solve_Analysis", "Solve_Strategy")
builder.add_edge("Solve_Strategy", "Solve_Computation")
builder.add_edge("Solve_Computation", END)

graph = builder.compile()
