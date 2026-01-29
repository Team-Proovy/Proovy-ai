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
from agents.workflows.utils import call_model
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
    """최종 한국어 응답을 생성하는 LangGraph 노드.

    - solve/explain/graph/variant/solution/check 등의 결과와
      review_state, suggestion_summary 등을 참고하여 자연스러운 한국어 답을 만든다.
    - 생성된 답변은 state["messages"]에 AIMessage로 추가되고,
      state["final_output"]["final_answer"]에 문자열로 저장된다.
    - LangGraph가 내부 LLM 호출을 자동 감지하여 토큰을 스트리밍합니다.
    """

    print("---MAIN: FINAL RESPONSE---")

    messages = state.get("messages") or []
    final_output = state.get("final_output") or {}
    review_state = state.get("review_state")

    # 최종 응답에 사용할 컨텍스트 구성
    user_text = _last_user_message(state) or ""

    # 너무 긴 JSON은 그대로 쓰되, 한국어 요약을 모델에 맡긴다.
    serialized_final = json.dumps(final_output, ensure_ascii=False, default=str)
    serialized_review = (
        json.dumps(review_state, ensure_ascii=False, default=str)
        if review_state is not None
        else ""
    )

    system_prompt = (
        "너는 수학·과학·프로그래밍 문제를 도와주는 한국어 튜터야. "
        "아래에 주어지는 사용자의 질문과 중간 계산/설명/리뷰 결과를 참고해서 "
        "사용자가 이해하기 쉬운 최종 답변을 한국어로 작성해 줘. "
        "너무 장황하지 않게 핵심 위주로 설명하고, 필요한 경우 2~4단계 정도의 "
        "간단한 풀이 과정을 포함해 줘."
    )

    # 모델에 건네줄 사용자 메시지
    parts: list[str] = []
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
    answer_text = call_model(MODEL_NAME, system_prompt, user_prompt, tags=[]).strip()
    ai_msg = AIMessage(content=answer_text)
    print("Final response : ", answer_text)
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
