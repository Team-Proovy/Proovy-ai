from typing import Dict, Any, List, Optional
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.llm import get_model
from core.settings import settings
from schema.models import OpenRouterModelName

from agents.prompts.review_prompts import REVIEW_SYSTEM_PROMPT, SUGGESTION_SYSTEM_PROMPT

MODEL_NAME = OpenRouterModelName.GPT_5_MINI


def _ensure_dict_review_state(review_state) -> Dict[str, Any]:
    if review_state is None:
        return {
            "passed": True,
            "feedback": None,
            "suggestions": [],
            "retry_count": 0,
            "reasons": [],
        }
    if isinstance(review_state, dict):
        return dict(review_state)
    try:
        return review_state.model_dump() if hasattr(review_state, "model_dump") else review_state.dict()
    except Exception:
        return {
            "passed": getattr(review_state, "passed", True),
            "feedback": getattr(review_state, "feedback", None),
            "suggestions": getattr(review_state, "suggestions", []),
            "retry_count": getattr(review_state, "retry_count", 0),
            "reasons": getattr(review_state, "reasons", []),
        }


def _serialize_model_obj(obj):
    try:
        return obj.model_dump(exclude_none=True) if hasattr(obj, "model_dump") else obj.dict(exclude_none=True)
    except Exception:
        return str(obj)


def _solution_next_suggestions(state: Dict[str, Any]) -> List[str]:
    progress = state.get("solution_progress")
    if isinstance(progress, dict):
        current = int(progress.get("current_chunk", 0) or 0)
        total = int(progress.get("total_chunks", 0) or 0)
        chunk_size = int(progress.get("chunk_size", 0) or 0)
        total_problems = int(progress.get("total_problems", 0) or 0)
    else:
        current = getattr(progress, "current_chunk", 0) if progress else 0
        total = getattr(progress, "total_chunks", 0) if progress else 0
        chunk_size = getattr(progress, "chunk_size", 0) if progress else 0
        total_problems = getattr(progress, "total_problems", 0) if progress else 0
    if total and current < total:
        remaining = max(0, int(total_problems or 0) - (int(current or 0) * int(chunk_size or 0)))
        next_batch = min(int(chunk_size or 0) or 5, remaining) if remaining else (int(chunk_size or 0) or 5)
        return [f"다음 {next_batch}문제도 풀어드릴까요?", "전체 요약본이 필요하신가요?"]

    final_output = state.get("final_output") or {}
    solution_view = final_output.get("solution") if isinstance(final_output, dict) else None
    if isinstance(solution_view, dict) and solution_view.get("next_chunk_available"):
        remaining = int(solution_view.get("remaining_count") or 0)
        chunk_size = int(solution_view.get("chunk_size") or 0) or 5
        next_batch = min(chunk_size, remaining) if remaining else chunk_size
        return [f"다음 {next_batch}문제도 풀어드릴까요?", "전체 요약본이 필요하신가요?"]
    return []


def _solve_next_suggestions(state: Dict[str, Any]) -> List[str]:
    problems = state.get("problems") or []
    if not isinstance(problems, list) or not problems:
        return []

    idx = int(state.get("current_problem_index", 0) or 0)
    idx = max(0, idx)
    total = len(problems)

    if idx + 1 < total:
        next_item = problems[idx + 1]
        next_number = (next_item or {}).get("number") if isinstance(next_item, dict) else None
        display_num = next_number if next_number is not None else (idx + 2)
        return [
            f"다음 {display_num}번 문제도 풀어드릴까요?",
            "방금 푼 문제를 다시 설명해드릴까요?",
        ]

    return ["전체 풀이를 요약해드릴까요?", "유사한 문제를 새로 만들어드릴까요?"]


def _diverse_followup_suggestions(state: Dict[str, Any]) -> List[str]:
    idx = int(state.get("current_problem_index", 0) or 0)
    pool = [
        "핵심 개념 3줄 요약으로 복습해볼까요?",
        "같은 유형 미니 퀴즈 2문제를 풀어볼까요?",
        "이번 문제의 실수 포인트를 체크해볼까요?",
        "풀이 전략을 한 단계씩 다시 정리해볼까요?",
    ]
    start = idx % len(pool)
    return [pool[start], pool[(start + 1) % len(pool)]]


def _solve_progress_context(state: Dict[str, Any]) -> Dict[str, Any]:
    problems = state.get("problems") or []
    if not isinstance(problems, list) or not problems:
        return {}
    idx = int(state.get("current_problem_index", 0) or 0)
    idx = max(0, min(idx, len(problems) - 1))
    current = idx + 1
    total = len(problems)
    current_number = None
    current_item = problems[idx]
    if isinstance(current_item, dict):
        current_number = current_item.get("number")
    next_number = None
    if current < total:
        next_item = problems[current]
        if isinstance(next_item, dict):
            next_number = next_item.get("number")
    return {
        "current_problem_order": current,
        "current_problem_number": current_number,
        "total_problems": total,
        "next_problem_number": next_number,
    }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*?\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _last_user_message(messages) -> Optional[str]:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return getattr(msg, "content", None) or str(msg)
    return None


def is_all_empty(feature_view: Dict[str, Any]) -> bool:
    for value in feature_view.values():
        if value is None:
            continue
        if isinstance(value, (dict, list, str)) and not value:
            continue
        serialized = _serialize_model_obj(value)
        if serialized not in (None, "", {}, [], "None"):
            return False
    return True


def rule_checks(feature_view: Dict[str, Any]) -> List[str]:
    reasons = []


    if is_all_empty(feature_view):
        reasons.append("no_step_result")
        return reasons

    solve = feature_view.get("solve_result")
    if solve:
        ans = solve.answer if hasattr(solve, "answer") else (solve.get("answer") if isinstance(solve, dict) else None)
        if not ans:
            reasons.append("empty_answer")

        comp = solve.computation if hasattr(solve, "computation") else (solve.get("computation") if isinstance(solve, dict) else None)
        if comp:
            success = getattr(comp, "success", None) if not isinstance(comp, dict) else comp.get("success")
            if success is False:
                reasons.append("computation_failed")

        steps = solve.steps if hasattr(solve, "steps") else (solve.get("steps") if isinstance(solve, dict) else [])
        if not steps:
            reasons.append("no_steps")
    else:
        explain = feature_view.get("explain_result")
        if explain:
            expl_text = getattr(explain, "explanation", "") if not isinstance(explain, dict) else explain.get("explanation", "")
            if len(str(expl_text).strip()) < 20:
                reasons.append("brief_explanation")
    return reasons


def run_review(state: Dict[str, Any]) -> Dict[str, Any]:
    feature_view = {
        "solve_result": state.get("solve_result"),
        "explain_result": state.get("explain_result"),
        "graph_result": state.get("graph_result"),
        "variant_result": state.get("variant_result"),
        "solution_result": state.get("solution_result"),
    }

    retry_count = state.get("retry_count", 0)
    reasons = rule_checks(feature_view)

    review_out = {
        "passed": True,
        "feedback": "All deterministic checks passed.",
        "suggestions": [],
        "retry_count": retry_count,
        "reasons": [],
    }

    if reasons:
        # Deterministic failure -> must fail. LLM only augments feedback & suggestions.
        review_out["passed"] = False
        review_out["reasons"] = reasons

        last_user_msg = _last_user_message(state.get("messages") or [])
        feature_summary = {k: (None if v is None else _serialize_model_obj(v)) for k, v in feature_view.items()}

        #모델 호출 로그 찍어보기기
        print(
            f"---REVIEW: MODEL={MODEL_NAME} openrouter_key_set={bool(settings.OPENROUTER_API_KEY)}---"
        )
        model = get_model(MODEL_NAME)
        prompt = [
            SystemMessage(content=REVIEW_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Detected deterministic issues: {reasons}\n"
                    f"Last user message: {last_user_msg}\n"
                    f"Feature summary: {json.dumps(feature_summary, ensure_ascii=False, default=str)}\n"
                    f"Please return JSON with keys feedback (short) and suggestions (list)."
                )
            ),
        ]

        parsed = None
        text = ""
        try:
            ai_msg = model.invoke(prompt)
            text = getattr(ai_msg, "content", "") or str(ai_msg)
            parsed = _extract_json(text)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            review_out["feedback"] = parsed.get("feedback", review_out["feedback"])
            review_out["suggestions"] = parsed.get("suggestions", [])
        else:
            review_out["feedback"] = text[:1000] if text else "Review evaluation failed."
            review_out["suggestions"] = []

        review_out["retry_count"] = retry_count

    return {"review_state": review_out}


def run_suggestion(state: Dict[str, Any]) -> Dict[str, Any]:
    print(
        f"---SUGGESTION: MODEL={MODEL_NAME} openrouter_key_set={bool(settings.OPENROUTER_API_KEY)}---"
    )
    model = get_model(MODEL_NAME)
    review_state = _ensure_dict_review_state(state.get("review_state"))
    last_user_msg = _last_user_message(state.get("messages") or [])

    feature_summary = {}
    for key in ("solve_result", "explain_result", "graph_result", "solution_result"):
        value = state.get(key)
        feature_summary[key] = None if value is None else _serialize_model_obj(value)
    solve_progress = _solve_progress_context(state)

    prompt = [
        SystemMessage(content=SUGGESTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Review: {json.dumps(review_state, ensure_ascii=False)}\n"
                f"Last user message: {last_user_msg}\n"
                f"Solve progress context: {json.dumps(solve_progress, ensure_ascii=False)}\n"
                f"Feature summary: {json.dumps(feature_summary, ensure_ascii=False, default=str)}\n"
                f"Return JSON with keys: ai_message, summary, suggestion_bullets."
            )
        ),
    ]

    parsed = None
    text = ""
    try:
        ai_msg = model.invoke(prompt)
        text = getattr(ai_msg, "content", "") or str(ai_msg)
        parsed = _extract_json(text)
    except Exception:
        parsed = None

    suggestion_json = parsed or {}
    suggestion_bullets = suggestion_json.get("suggestion_bullets", [])
    ai_message_text = suggestion_json.get("ai_message", text or "다음 학습 방향을 제안합니다.")
    summary = suggestion_json.get("summary", suggestion_bullets)

    extra_suggestions = _solution_next_suggestions(state)
    solve_suggestions = _solve_next_suggestions(state)
    for item in solve_suggestions:
        if item not in extra_suggestions:
            extra_suggestions.append(item)
    for item in _diverse_followup_suggestions(state):
        if item not in extra_suggestions:
            extra_suggestions.append(item)

    if solve_progress and solve_progress.get("total_problems", 0) > 1:
        current = solve_progress.get("current_problem_order")
        total = solve_progress.get("total_problems")
        current_number = solve_progress.get("current_problem_number")
        if current_number is not None:
            progress_prefix = (
                f"현재 {total}문제 중 {current}번째(문항 {current_number}번) 문제를 진행 중입니다."
            )
        else:
            progress_prefix = f"현재 {total}문제 중 {current}번째 문제를 진행 중입니다."
        if progress_prefix not in ai_message_text:
            ai_message_text = f"{progress_prefix}\n{ai_message_text}"

    if extra_suggestions:
        if not isinstance(suggestion_bullets, list):
            suggestion_bullets = []
        # LLM 제안의 "다음 문제" 계열은 시스템 제안과 충돌하므로 제거한다.
        next_problem_pattern = re.compile(r"다음\s*(?:\d+\s*번\s*)?문제")
        suggestion_bullets = [
            item for item in suggestion_bullets
            if isinstance(item, str) and not next_problem_pattern.search(item)
        ]
        for item in extra_suggestions:
            if item not in suggestion_bullets:
                suggestion_bullets.append(item)
        # 첫 제안은 항상 "다음 N번 문제..."를 우선 배치하며, 전체 개수를 4개로 제한한다.
        next_candidates = [
            item for item in suggestion_bullets
            if isinstance(item, str) and re.search(r"다음\s*\d+\s*번\s*문제", item)
        ]
        if next_candidates:
            first_next = next_candidates[0]
            others = [item for item in suggestion_bullets if item != first_next]
            suggestion_bullets = [first_next] + others[:3]
        else:
            suggestion_bullets = suggestion_bullets[:4]

    if suggestion_bullets and ai_message_text:
        bullet_lines = "\n".join(f"- {item}" for item in suggestion_bullets)
        ai_message_text = f"{ai_message_text}\n\n다음 학습 제안:\n{bullet_lines}"
    elif suggestion_bullets:
        bullet_lines = "\n".join(f"- {item}" for item in suggestion_bullets)
        ai_message_text = f"다음 학습 제안:\n{bullet_lines}"

    try:
        ai_message_obj = AIMessage(content=ai_message_text)
    except Exception:
        ai_message_obj = {"role": "assistant", "content": ai_message_text}

    current_final = state.get("final_output") or {}
    updated_final = dict(current_final) if isinstance(current_final, dict) else {"text": str(current_final)}
    # 성공 케이스에서도 review를 항상 포함하도록 보장
    if "review" not in updated_final:
        updated_final["review"] = review_state
    updated_final["suggestion_summary"] = summary
    if suggestion_bullets:
        updated_final["suggestion_bullets"] = suggestion_bullets

    return {"messages": [ai_message_obj], "final_output": updated_final}

