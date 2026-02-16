"""Final response node for Proovy agent.

이 노드는 전체 그래프 실행이 끝난 뒤, 축적된 중간 결과(final_output 등)를
바탕으로 사용자에게 보여줄 최종 한국어 답변을 생성합니다.

- 입력: AgentState (messages, final_output, review_state 등 포함)
- 출력: AgentState (messages에 최종 AIMessage 추가, final_output에 final_answer 반영)
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from agents.prompts.final_response_prompts import (
    PARTIAL_RESPONSE_SYSTEM_PROMPT,
    PARTIAL_RESPONSE_USER_INSTRUCTION,
    TRADITIONAL_RESPONSE_SYSTEM_PROMPT,
    TRADITIONAL_RESPONSE_USER_INSTRUCTION,
)
from agents.state import AgentState
from agents.workflows.utils import call_model, get_conversation_summary
from schema.models import OpenRouterModelName


MODEL_NAME = OpenRouterModelName.GPT_5_MINI


def _format_review_message(review_state: dict | None) -> str:
    """review_state를 사람이 읽기 좋은 한국어 메시지로 변환합니다.

    JSON 형태로 LLM에 전달하면 응답에 그대로 노출되는 문제가 있어,
    자연스러운 문장 형태로 변환하여 전달합니다.
    """
    if not review_state:
        return ""

    if isinstance(review_state, str):
        return review_state

    parts: list[str] = []

    # 피드백
    feedback = review_state.get("feedback")
    if feedback and feedback != "All deterministic checks passed.":
        parts.append(f"검토 결과: {feedback}")

    # 제안사항
    suggestions = review_state.get("suggestions")
    if suggestions and isinstance(suggestions, list) and any(suggestions):
        parts.append("개선 제안:")
        for i, suggestion in enumerate(suggestions, 1):
            if suggestion:
                parts.append(f"  {i}. {suggestion}")

    # 검토 사유
    reasons = review_state.get("reasons")
    if reasons and isinstance(reasons, list) and any(reasons):
        parts.append(f"검토 사유: {', '.join(reasons)}")

    return "\n".join(parts) if parts else ""


def _last_user_message(state: AgentState) -> str | None:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None)
        if msg_type in {"human", "user"}:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                text = content.strip()
            else:
                # list/dict 등 복합 콘텐츠는 문자열로 단순 직렬화
                text = str(content).strip()
            if text:
                return text
    return None


def final_response(state: AgentState) -> AgentState:
    """최종 한국어 응답을 생성하는 LangGraph 노드 (하이브리드 방식).

    - partial_responses가 있으면: Writer 노드들이 생성한 부분 응답을 조합하고,
      Suggestion만 추가로 생성하여 결합 (토큰 절약)
    - partial_responses가 없으면: 기존 방식대로 전체 final_output을 LLM으로 종합
    - 생성된 답변은 state["messages"]에 AIMessage로 추가되고,
      state["final_output"]["final_answer"]에 문자열로 저장된다.
    - LangGraph가 내부 LLM 호출을 자동 감지하여 토큰을 스트리밍합니다.
    """

    print("---MAIN: FINAL RESPONSE---")

    messages = state.get("messages") or []
    final_output = state.get("final_output") or {}
    review_state = state.get("review_state")
    partial_responses = state.get("partial_responses", [])

    # === 하이브리드 방식: partial_responses 우선 사용 ===
    # 주의: UI가 /stream의 token 이벤트만 렌더링하는 경우를 위해,
    # partial_responses가 있어도 최종 문장은 반드시 LLM 호출로 생성한다.
    if partial_responses:
        print(f"Using {len(partial_responses)} partial response(s) from Writer nodes")

        # Writer 노드들이 생성한 부분들을 구조화해서 전달
        feature_sections: list[str] = []
        for pr in partial_responses:
            feature = pr.get("feature", "Unknown")
            content = pr.get("content", "")
            if content:
                feature_sections.append(f"[{feature}]\n{content}")

        # Suggestion이 있으면 추가
        suggestion_summary = final_output.get("suggestion_summary")
        suggestion_bullets = final_output.get("suggestion_bullets")

        user_text = _last_user_message(state) or ""

        prompt_parts: list[str] = []
        if user_text:
            prompt_parts.append(f"[사용자 질문]\n{user_text}")
        if feature_sections:
            prompt_parts.append(
                "[Writer 부분 응답들]\n" + "\n\n---\n\n".join(feature_sections)
            )
        if isinstance(suggestion_bullets, list) and suggestion_bullets:
            bullets = "\n".join(f"- {item}" for item in suggestion_bullets)
            prompt_parts.append(f"[다음 학습 제안]\n{bullets}")
        elif suggestion_summary:
            prompt_parts.append(f"[다음 학습 제안 요약]\n{suggestion_summary}")

        prompt_parts.append(PARTIAL_RESPONSE_USER_INSTRUCTION)

        system_prompt = PARTIAL_RESPONSE_SYSTEM_PROMPT
        user_prompt = "\n\n".join(prompt_parts)
        answer_text = call_model(
            MODEL_NAME, system_prompt, user_prompt, tags=[]
        ).strip()
        print(
            f"Final response from partial_responses via LLM (length: {len(answer_text)})"
        )

    else:
        # === 기존 방식: 전체 final_output을 LLM으로 종합 ===
        print("No partial_responses, using traditional full LLM synthesis")

        user_text = _last_user_message(state) or ""

        # OCR 원문(problem)은 응답에 포함하지 않음 - 핵심 결과만 전달
        filtered_final = {}
        if isinstance(final_output, dict):
            # OCR 원문 및 review JSON 관련 필드 제외 (review는 별도 포맷팅)
            exclude_keys = {"problem", "ocr_text", "ocr_blocks", "raw_ocr", "review"}
            for k, v in final_output.items():
                if k not in exclude_keys:
                    filtered_final[k] = v
        else:
            filtered_final = {"raw": str(final_output)}

        serialized_final = json.dumps(filtered_final, ensure_ascii=False, default=str)
        # review_state는 JSON이 아닌 자연스러운 한국어 메시지로 변환
        formatted_review = _format_review_message(review_state)

        # 이전 대화 맥락 수집 (멀티턴 대화 지원)
        conversation_context = get_conversation_summary(state, max_chars=1500)

        system_prompt = TRADITIONAL_RESPONSE_SYSTEM_PROMPT

        # 모델에 건네줄 사용자 메시지
        parts: list[str] = []
        if conversation_context:
            parts.append(f"[이전 대화 기록]\n{conversation_context}")
        if user_text:
            parts.append(f"[사용자 질문]\n{user_text}")
        if serialized_final:
            parts.append(f"[중간 결과 요약(final_output)]\n{serialized_final}")
        if formatted_review:
            parts.append(f"[검토 결과]\n{formatted_review}")

        parts.append(TRADITIONAL_RESPONSE_USER_INSTRUCTION)

        user_prompt = "\n\n".join(parts)

        # 공통 유틸리티 함수를 사용해 LLM을 호출한다.
        # 기본 call_model은 tags=["skip_stream"]로 토큰 스트리밍을 건너뛰지만,
        # 최종 응답 노드는 토큰을 스트리밍해야 하므로 tags=[]로 덮어쓴다.
        answer_text = call_model(
            MODEL_NAME, system_prompt, user_prompt, tags=[]
        ).strip()
        print("Final response (traditional): ", answer_text)

    # === AIMessage 생성 및 state 업데이트 ===
    ai_msg = AIMessage(content=answer_text)

    # LangGraph state에 AIMessage 추가
    if not isinstance(ai_msg, AIMessage):
        content = getattr(ai_msg, "content", "")
        ai_msg = AIMessage(content=content)

    state["messages"] = (messages or []) + [ai_msg]

    # final_output에 최종 답변 문자열 저장
    if isinstance(final_output, dict):
        final_output = dict(final_output)
    else:
        final_output = {"raw": str(final_output)}

    final_output["final_answer"] = answer_text
    state["final_output"] = final_output
    state["prev_action"] = "FinalResponse"

    return state
