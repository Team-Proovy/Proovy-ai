from typing import Dict, Any, List, Optional, Union, Tuple
import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from core.llm import get_model
from core.settings import settings
from schema.models import OpenRouterModelName

from agents.prompts.review_prompts import REVIEW_SYSTEM_PROMPT, SUGGESTION_SYSTEM_PROMPT
from agents.workflows.subgraphs.step4_features.solution.problem_utils import extract_problem_number

MODEL_NAME = OpenRouterModelName.GPT_5_MINI


# [개선] PII 정규표현식 미리 컴파일하여 성능 최적화
PII_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+|010-\d{4}-\d{4}")


def mask_pii(text: Any) -> Tuple[bool, str]:
    """텍스트 내 개인정보(PII)를 탐지하고 마스킹합니다."""
    if not isinstance(text, str):
        text = str(text or "")
    if not text:
        return False, ""
    found = bool(PII_REGEX.search(text))
    masked = PII_REGEX.sub("[PII]", text)
    return found, masked


class SuggestionItem(BaseModel):
    """개별 제안 항목의 구조"""
    text: str = Field(description="제안 내용")
    type: str = Field(default="recommend", description="제안 유형 (practice, review, challenge, etc.)")
    priority: int = Field(default=3, description="우선순위 (1: 높음, 5: 낮음)")
    source: str = Field(default="llm", description="제안 출처 (auto, llm)")


class SuggestionSchema(BaseModel):
    """지능형 학습 제안을 위한 구조화된 데이터 스키마"""
    ai_message: str = Field(description="사용자에게 보여줄 친절한 한국어 안내 메시지 (이모지 제외)")
    summary: str = Field(description="제안의 요약 (최대 200자)")
    suggestion_bullets: List[SuggestionItem] = Field(default_factory=list, description="구체적인 학습 제안 항목 리스트 (2~3개 권장)")
    confidence: float = Field(default=1.0, description="제안의 신뢰도 (0.0~1.0)")
    pii_detected: bool = Field(default=False, description="개인정보(PII) 포함 여부")


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
    if isinstance(obj, dict):
        return obj
    try:
        return obj.model_dump(exclude_none=True) if hasattr(obj, "model_dump") else obj.dict(exclude_none=True)
    except Exception:
        return str(obj)


def _solution_next_suggestions(state: Dict[str, Any]) -> List[str]:
    """다음 학습 단계를 결정하는 Single Source of Truth 로직 (마지막 문제 대응)"""
    last_idx = state.get("last_solved_index")
    chunks = state.get("solution_chunks") or []
    progress = state.get("solution_progress")
    
    # [개선] 다음 인덱스 안전 계산 로직 (0-based)
    next_idx = (last_idx if last_idx is not None else -1) + 1
    total_count = len(chunks)
    
    # 1. last_solved_index 기반 제안 (우선순위)
    if chunks and 0 <= next_idx < total_count:
        remaining_problems = chunks[next_idx:]
        remaining_count = len(remaining_problems)
        next_problem_text = remaining_problems[0]
        next_num = extract_problem_number(next_problem_text)
        
        suggestion_text = f"다음 {next_num}번 문제도 풀어드릴까요?" if next_num else f"다음 {remaining_count}문제도 풀어드릴까요?"
        if not next_num and remaining_count == 1:
            suggestion_text = "다음 문제도 풀어드릴까요?"
            
        suggestions = [suggestion_text, "전체 요약본이 필요하신가요?"]
        if remaining_count > 1:
            # [개선] 일괄 풀기 옵션을 더 구체적으로 표기
            estimated_time = remaining_count * 10 # 대략 문제당 10초 가정
            suggestions.insert(1, f"남은 {remaining_count}문제 모두 풀기 (약 {estimated_time}초 소요)")
        return suggestions

    # 2. 마지막 문제까지 모두 푼 경우 (다양한 마무리 제안)
    if chunks and last_idx is not None and last_idx >= total_count - 1:
        return [
            "오늘 푼 문제들의 전체 요약본 PDF 만들기",
            "방금 푼 유형의 심화 문제 도전하기",
            "핵심 개념 다시 정리하기"
        ]

    # 3. progress 기반 폴백 (기존 Solution 로직 호환)
    if progress:
        # ... (기존 progress 로직 유지)
        if isinstance(progress, dict):
            current = int(progress.get("current_chunk", 0) or 0)
            total = int(progress.get("total_chunks", 0) or 0)
            chunk_size = int(progress.get("chunk_size", 0) or 0)
        else:
            current = getattr(progress, "current_chunk", 0)
            total = getattr(progress, "total_chunks", 0)
            chunk_size = getattr(progress, "chunk_size", 0)

        if total and current < total:
            start_idx = current * chunk_size
            remaining_problems = chunks[start_idx:]
            if remaining_problems:
                next_problem_text = remaining_problems[0]
                next_num = extract_problem_number(next_problem_text)
                suggestion_text = f"다음 {next_num}번 문제도 풀어드릴까요?" if next_num else "다음 문제도 풀어드릴까요?"
                return [suggestion_text, "전체 요약본이 필요하신가요?"]

    # [수정] Unreachable Code 제거 및 state 기반 폴백 로직 통합
    final_output = state.get("final_output", {})
    solution_view = final_output.get("solution") if isinstance(final_output, dict) else None
    if isinstance(solution_view, dict) and solution_view.get("next_chunk_available"):
        remaining = int(solution_view.get("remaining_count") or 0)
        return [f"다음 {remaining}문제도 풀어드릴까요?", "전체 요약본이 필요하신가요?"]
        
    return []


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """텍스트에서 JSON 블록을 추출하여 파싱합니다. (중첩 및 노이즈 대응)"""
    if not text:
        return None
    cleaned = text.strip()
    try:
        # 1. 전체가 바로 JSON인 경우
        return json.loads(cleaned)
    except Exception:
        # 2. Markdown 코드 블록 제거 시도
        if cleaned.startswith("```"):
            segments = []
            for part in cleaned.split("```"):
                part = part.strip()
                if not part or part.lower().startswith("json"):
                    continue
                segments.append(part)
            cleaned = "\n".join(segments).strip()
            try:
                return json.loads(cleaned)
            except Exception:
                pass
        
        # 3. 가장 바깥쪽 { } 찾기
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                # 4. 노이즈 제거 시도 (따옴표 등 간단한 수정)
                try:
                    return json.loads(candidate.replace("'", '"'))
                except Exception:
                    return None
    return None


def _last_user_message(messages) -> Optional[str]:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(text_parts).strip()
            return str(content or "")
    return None


def safe_invoke_model(model, prompt, timeout=15, retries=2) -> Any:
    """모델 호출 시 타임아웃 및 재시도 로직 적용. 반환값은 모델 설정에 따라 다름."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            # LangChainChat 모델의 invoke 호출
            return model.invoke(prompt, timeout=timeout)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
    
    print(f"safe_invoke_model failed after {retries} retries: {last_exc}")
    return None


def record_diagnostics(state: Dict[str, Any], raw: Any, parsed_ok: bool, parse_err: Optional[str], duration_ms: int):
    """최종 출력에 진단 정보를 기록합니다."""
    final_output = state.setdefault("final_output", {})
    diag = final_output.get("diagnostics", {})
    
    # 원본 응답 저장 (문자열 변환 및 길이 제한)
    raw_str = str(raw) if raw else ""
    diag.update({
        "last_suggestion": {
            "raw_output": raw_str[:2000] + ("..." if len(raw_str) > 2000 else ""),
            "parsed_ok": parsed_ok,
            "parse_error": parse_err,
            "duration_ms": duration_ms,
            "timestamp": time.time()
        }
    })
    final_output["diagnostics"] = diag


def invoke_and_parse_suggestion(base_model, prompt) -> Tuple[Optional[SuggestionSchema], Any, int, Optional[str]]:
    """통합 호출, 파싱, 시간 측정을 수행합니다."""
    start_time = time.time()
    structured_model = None
    try:
        # Pydantic 모델을 통한 구조화된 출력 시도 (LangChain 지원 기능)
        structured_model = base_model.with_structured_output(SuggestionSchema)
    except Exception as e:
        print(f"with_structured_output not supported or failed: {e}")

    raw_response = None
    parsed_obj: Optional[SuggestionSchema] = None
    parse_err = None
    
    try:
        if structured_model:
            # 구조화된 모델 호출
            raw_response = safe_invoke_model(structured_model, prompt)
            
            if isinstance(raw_response, SuggestionSchema):
                parsed_obj = raw_response
            elif isinstance(raw_response, dict):
                parsed_obj = SuggestionSchema(**raw_response)
            elif hasattr(raw_response, "content"): # AIMessage 등
                text = getattr(raw_response, "content", "")
                parsed = _extract_json(text)
                if parsed:
                    parsed_obj = SuggestionSchema(**parsed)
        
        # 구조화 호출에 실패했거나 지원하지 않는 경우 일반 호출 시도
        if not parsed_obj:
            raw_response = safe_invoke_model(base_model, prompt)
            text = getattr(raw_response, "content", "") if hasattr(raw_response, "content") else str(raw_response)
            parsed = _extract_json(text)
            if parsed:
                parsed_obj = SuggestionSchema(**parsed)
                
    except Exception as e:
        parse_err = str(e)
        print(f"Error in invoke_and_parse_suggestion: {e}")

    duration_ms = int((time.time() - start_time) * 1000)
    return parsed_obj, raw_response, duration_ms, parse_err


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
            # [개선] 일관된 호출 방식 적용
            ai_msg = safe_invoke_model(model, prompt)
            text = getattr(ai_msg, "content", "") if hasattr(ai_msg, "content") else str(ai_msg)
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
    base_model = get_model(MODEL_NAME)
    
    review_state = _ensure_dict_review_state(state.get("review_state"))
    last_user_msg = _last_user_message(state.get("messages") or [])

    feature_summary = {}
    for key in ("solve_result", "explain_result", "graph_result", "solution_result", "solution_chunks"):
        value = state.get(key)
        feature_summary[key] = None if value is None else _serialize_model_obj(value)

    # [PII 전처리] 입력 텍스트 마스킹
    pii_found_user, masked_user_msg = mask_pii(last_user_msg)
    pii_found_feat, masked_feat_summary = mask_pii(json.dumps(feature_summary, ensure_ascii=False, default=str))

    # [개선] 진행 상황 정보를 프롬프트에 명확히 주입하여 오판 방지
    last_idx = state.get("last_solved_index")
    chunks = state.get("solution_chunks") or []
    progress_hint = f"현재 전체 {len(chunks)}문제 중 {last_idx + 1}번째 문제까지 풀이 완료." if last_idx is not None else "아직 풀이 시작 전입니다."
    
    if last_idx is None:
        next_step_hint = "첫 번째 문제를 풀 차례입니다." if chunks else "새로운 문제를 풀 차례입니다."
    elif (last_idx + 1) < len(chunks):
        next_step_hint = f"다음은 {last_idx + 2}번 문제를 풀 차례입니다."
    else:
        next_step_hint = "모든 문제를 풀었습니다."

    prompt = [
        SystemMessage(content=SUGGESTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Learning Progress: {progress_hint} {next_step_hint}\n"
                f"Review: {json.dumps(review_state, ensure_ascii=False)}\n"
                f"Last user message: {masked_user_msg}\n"
                f"Feature summary: {masked_feat_summary}\n"
                f"Return JSON matching SuggestionSchema."
            )
        ),
    ]

    # [개선] 통합 호출, 파싱, 진단 정보 수집
    suggestion_struct, raw_resp, duration, parse_err = invoke_and_parse_suggestion(base_model, prompt)

    # 기본값 설정 (Fallback)
    ai_message_text = "다음 학습 방향을 제안합니다."
    summary = "학습 제안 생성"
    final_items: List[SuggestionItem] = []
    seen_texts = set()

    # 1. 자동 생성된 제안(남은 문제 등) 추가 (최우선순위, source="auto")
    extra_suggestions = _solution_next_suggestions(state)
    for text in extra_suggestions:
        if text not in seen_texts:
            item = SuggestionItem(text=text, type="practice", priority=1, source="auto")
            final_items.append(item)
            seen_texts.add(text)
            
    # 2. 모델이 생성한 제안 추가
    if suggestion_struct:
        ai_message_text = suggestion_struct.ai_message
        summary = suggestion_struct.summary
        for llm_item in suggestion_struct.suggestion_bullets:
            if llm_item.text not in seen_texts:
                llm_item.source = "llm" # 명시적 출처 설정
                final_items.append(llm_item)
                seen_texts.add(llm_item.text)

    # [규칙 적용] 이모지 제거 (내부 규칙 준수)
    ai_message_text = str(ai_message_text or "")
    ai_message_text = re.sub(r'[^\w\s가-힣\d\.,!\?\(\)\-\:\[\]]', '', ai_message_text).strip()

    # 우선순위 정렬 및 길이 제한 적용 (최대 5개)
    final_items.sort(key=lambda x: x.priority)
    final_items = final_items[:5]

    if final_items:
        bullet_lines = []
        for item in final_items:
            text_val = ""
            # item이 SuggestionItem 객체인 경우
            if hasattr(item, "text") and not isinstance(item, dict):
                text_val = item.text
            # item이 딕셔너리인 경우
            elif isinstance(item, dict) and "text" in item:
                text_val = item["text"]
            # 그 외 (문자열 등)
            else:
                text_val = str(item)
            
            if text_val:
                bullet_lines.append(f"- {text_val}")
        
        if bullet_lines:
            ai_message_text = f"{ai_message_text}\n\n**다음 학습 제안:**\n" + "\n".join(bullet_lines)

    try:
        ai_message_obj = AIMessage(content=ai_message_text)
    except Exception:
        ai_message_obj = {"role": "assistant", "content": ai_message_text}

    # 진단 정보 기록
    record_diagnostics(state, raw_resp, suggestion_struct is not None, parse_err, duration)

    # [개선] 요청한 타겟 문제가 실제로 해결되었는지 확인 후 클리어
    target = state.get("target_problem_number")
    if target is not None:
        chunks = state.get("solution_chunks") or []
        for i, chunk in enumerate(chunks):
            if extract_problem_number(chunk) == target:
                if state.get("last_solved_index") == i:
                    state.pop("target_problem_number", None)
                    print(f"---SUGGESTION: TARGET PROBLEM {target} CLEARED AFTER RESOLUTION---")
                break

    current_final = state.get("final_output") or {}
    updated_final = dict(current_final) if isinstance(current_final, dict) else {"text": str(current_final)}
    
    if "review" not in updated_final:
        updated_final["review"] = review_state
    
    updated_final["suggestion_summary"] = summary
    # [수정] Pydantic 객체와 문자열/딕셔너리가 섞여 있어도 안전하게 직렬화
    if final_items:
        updated_final["suggestion_bullets"] = [
            item.model_dump() if hasattr(item, "model_dump") else {"text": str(item)} 
            for item in final_items
        ]
    
    if suggestion_struct and suggestion_struct.pii_detected:
        updated_final["pii_detected"] = True
        
    # [추가 보안] AI 응답 텍스트 내 PII 여부 다시 한번 체크 (컴파일된 REGEX 사용)
    if PII_REGEX.search(ai_message_text):
        updated_final["pii_detected"] = True

    return {"messages": [ai_message_obj], "final_output": updated_final}

