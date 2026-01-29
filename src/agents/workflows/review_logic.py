from typing import Dict, Any, List, Optional
import json
import re

from langchain_core.messages import AIMessage, HumanMessage
from agents.workflows.utils import call_model
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
        return (
            review_state.model_dump()
            if hasattr(review_state, "model_dump")
            else review_state.dict()
        )
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
        return (
            obj.model_dump(exclude_none=True)
            if hasattr(obj, "model_dump")
            else obj.dict(exclude_none=True)
        )
    except Exception:
        return str(obj)


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
        ans = (
            solve.answer
            if hasattr(solve, "answer")
            else (solve.get("answer") if isinstance(solve, dict) else None)
        )
        if not ans:
            reasons.append("empty_answer")

        comp = (
            solve.computation
            if hasattr(solve, "computation")
            else (solve.get("computation") if isinstance(solve, dict) else None)
        )
        if comp:
            success = (
                getattr(comp, "success", None)
                if not isinstance(comp, dict)
                else comp.get("success")
            )
            if success is False:
                reasons.append("computation_failed")

        steps = (
            solve.steps
            if hasattr(solve, "steps")
            else (solve.get("steps") if isinstance(solve, dict) else [])
        )
        if not steps:
            reasons.append("no_steps")
    else:
        explain = feature_view.get("explain_result")
        if explain:
            expl_text = (
                getattr(explain, "explanation", "")
                if not isinstance(explain, dict)
                else explain.get("explanation", "")
            )
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
        feature_summary = {
            k: (None if v is None else _serialize_model_obj(v))
            for k, v in feature_view.items()
        }

        # 모델 호출 로그 찍어보기기

        print(
            f"---REVIEW: MODEL={MODEL_NAME} openrouter_key_set={bool(settings.OPENROUTER_API_KEY)}---"
        )

        user_prompt = (
            f"Detected deterministic issues: {reasons}\n"
            f"Last user message: {last_user_msg}\n"
            f"Feature summary: {json.dumps(feature_summary, ensure_ascii=False, default=str)}\n"
            f"Please return JSON with keys feedback (short) and suggestions (list)."
        )

        parsed = None
        text = ""
        try:
            text = call_model(MODEL_NAME, REVIEW_SYSTEM_PROMPT, user_prompt)
            parsed = _extract_json(text)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            review_out["feedback"] = parsed.get("feedback", review_out["feedback"])
            review_out["suggestions"] = parsed.get("suggestions", [])
        else:
            review_out["feedback"] = (
                text[:1000] if text else "Review evaluation failed."
            )
            review_out["suggestions"] = []

        review_out["retry_count"] = retry_count

    return {"review_state": review_out}


def run_suggestion(state: Dict[str, Any]) -> Dict[str, Any]:
    print(
        f"---SUGGESTION: MODEL={MODEL_NAME} openrouter_key_set={bool(settings.OPENROUTER_API_KEY)}---"
    )
    review_state = _ensure_dict_review_state(state.get("review_state"))
    last_user_msg = _last_user_message(state.get("messages") or [])

    feature_summary = {}
    for key in ("solve_result", "explain_result", "graph_result"):
        value = state.get(key)
        feature_summary[key] = None if value is None else _serialize_model_obj(value)

    user_prompt = (
        f"Review: {json.dumps(review_state, ensure_ascii=False)}\n"
        f"Last user message: {last_user_msg}\n"
        f"Feature summary: {json.dumps(feature_summary, ensure_ascii=False, default=str)}\n"
        f"Return JSON with keys: ai_message, summary, suggestion_bullets."
    )

    parsed = None
    text = ""
    try:
        text = call_model(MODEL_NAME, SUGGESTION_SYSTEM_PROMPT, user_prompt)
        parsed = _extract_json(text)
    except Exception:
        parsed = None

    suggestion_json = parsed or {}
    suggestion_bullets = suggestion_json.get("suggestion_bullets", [])
    ai_message_text = suggestion_json.get(
        "ai_message", text or "다음 학습 방향을 제안합니다."
    )
    summary = suggestion_json.get("summary", suggestion_bullets)

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
    updated_final = (
        dict(current_final)
        if isinstance(current_final, dict)
        else {"text": str(current_final)}
    )
    # 성공 케이스에서도 review를 항상 포함하도록 보장
    if "review" not in updated_final:
        updated_final["review"] = review_state
    updated_final["suggestion_summary"] = summary
    if suggestion_bullets:
        updated_final["suggestion_bullets"] = suggestion_bullets

    return {"messages": [ai_message_obj], "final_output": updated_final}
