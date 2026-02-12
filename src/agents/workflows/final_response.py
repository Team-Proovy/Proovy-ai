"""Final response node for Proovy agent.

이 노드는 전체 그래프 실행이 끝난 뒤, 축적된 중간 결과(final_output 등)를
바탕으로 사용자에게 보여줄 최종 한국어 답변을 생성합니다.

- 입력: AgentState (messages, final_output, review_state 등 포함)
- 출력: AgentState (messages에 최종 AIMessage 추가, final_output에 final_answer 반영)
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from agents.state import AgentState
from agents.workflows.utils import call_model, get_conversation_summary
from schema.models import OpenRouterModelName


MODEL_NAME = OpenRouterModelName.GPT_5_MINI


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
        conversation_context = get_conversation_summary(state, max_chars=1500)

        prompt_parts: list[str] = []
        if conversation_context:
            prompt_parts.append(f"[이전 대화 기록]\n{conversation_context}")
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

        prompt_parts.append(
            "위 정보를 바탕으로 사용자에게 보여줄 최종 한국어 답변을 작성해 줘. "
            "핵심 풀이를 간결하게 정리하고, 마지막에는 '다음 학습 제안' 섹션을 반드시 포함해 줘. "
            "수식은 필요할 때만 간단한 LaTeX로 표기해도 좋아."
        )

        system_prompt = (
            "너는 수학·과학·프로그래밍 문제를 도와주는 한국어 튜터야. "
            "주어진 부분 응답들을 중복 없이 자연스럽게 통합하고, "
            "사용자가 다음에 무엇을 공부하면 좋은지 명확히 안내해 줘."
        )
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
            # OCR 원문 관련 필드 제외
            exclude_keys = {"problem", "ocr_text", "ocr_blocks", "raw_ocr"}
            for k, v in final_output.items():
                if k not in exclude_keys:
                    filtered_final[k] = v
        else:
            filtered_final = {"raw": str(final_output)}

        serialized_final = json.dumps(filtered_final, ensure_ascii=False, default=str)
        serialized_review = (
            json.dumps(review_state, ensure_ascii=False, default=str)
            if review_state is not None
            else ""
        )

        # 이전 대화 맥락 수집 (멀티턴 대화 지원)
        conversation_context = get_conversation_summary(state, max_chars=1500)

        system_prompt = (
            "너는 수학·과학·프로그래밍 문제를 도와주는 한국어 튜터야. "
            "아래에 주어지는 사용자의 질문과 중간 계산/설명/리뷰 결과를 참고해서 "
            "사용자가 이해하기 쉬운 최종 답변을 한국어로 작성해 줘. "
            "너무 장황하지 않게 핵심 위주로 설명하고, 필요한 경우 2~4단계 정도의 "
            "간단한 풀이 과정을 포함해 줘. "
            "사용자가 이전 대화를 참조하는 경우(예: '이전 문제', '방금 푼 문제'), "
            "대화 기록을 참고하여 적절히 답변해 줘."
        )

        # 모델에 건네줄 사용자 메시지
        parts: list[str] = []
        if conversation_context:
            parts.append(f"[이전 대화 기록]\n{conversation_context}")
        if user_text:
            parts.append(f"[사용자 질문]\n{user_text}")
        if serialized_final:
            parts.append(f"[중간 결과 요약(final_output)]\n{serialized_final}")
        if serialized_review:
            parts.append(f"[리뷰/재시도 정보(review_state)]\n{serialized_review}")

        parts.append(
            "위 정보를 종합해서, 사용자에게 보여줄 최종 한국어 답변을 작성해 줘. "
            "답변은 친절하지만 불필요하게 길지 않게 하고, 수식이 있다면 LaTeX 형태로 간단히 표기해도 좋아."
        )

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
