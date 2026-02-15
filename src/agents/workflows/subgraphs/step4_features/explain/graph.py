"""Explain feature subgraph.

단일 노드에서 사용자의 마지막 질문을 간단히 풀어 설명하는 용도이다.
난이도에 따라 적절한 LLM 모델을 선택하여 호출한다.

난이도별 모델:
- easy: Gemini 2.5 Flash
- medium: Gemini 3 Flash
- hard: Gemini 3 Pro

checkpointer가 저장한 대화 히스토리를 활용하여 멀티턴 대화를 지원한다.
"""

from langgraph.graph import END, StateGraph

from agents.state import AgentState, ExplainResult
from agents.workflows.utils import (
    call_model_by_difficulty,
    classify_difficulty,
    extract_ocr_text,
    get_conversation_summary,
    get_difficulty_from_state,
    recent_user_context,
    set_difficulty_in_state,
)


def explain(state: AgentState) -> AgentState:
    print("---FEATURE: EXPLAIN---")

    # 최근 사용자 메시지를 가져온다 (대화 맥락 포함)
    user_text = recent_user_context(state, max_messages=3, include_assistant=True)
    explain_result = state.get("explain_result") or ExplainResult()

    # 난이도 분류 (Explain 단계에서 직접 수행)
    ocr_text = extract_ocr_text(state)
    combined_question = user_text or ""
    if ocr_text:
        combined_question = f"{combined_question}\n{ocr_text}".strip()
    difficulty = classify_difficulty(combined_question)
    set_difficulty_in_state(state, difficulty)
    print(f"---EXPLAIN: DIFFICULTY CLASSIFICATION RESULT {difficulty}---")

    # 이전 대화 맥락 수집 (멀티턴 지원)
    conversation_context = get_conversation_summary(state, max_chars=1000)

    if user_text:
        # 대화 맥락이 있으면 시스템 프롬프트에 포함
        context_info = ""
        if conversation_context:
            context_info = (
                "\n\nConsider the previous conversation context when explaining. "
                "If user refers to previous problems or topics, use that context."
            )

        system_prompt = (
            "You are a kind Korean tutor. "
            "Explain the given concept or question in very simple Korean, "
            "using short sentences and, if helpful, 1-2 easy examples."
            f"{context_info}"
        )

        # 대화 맥락이 있으면 프롬프트에 포함
        history_section = ""
        if conversation_context:
            history_section = f"\n\n[이전 대화 기록]\n{conversation_context}\n"

        user_prompt = (
            f"{history_section}"
            f"사용자 질문 또는 개념:\n{user_text}\n\n"
            "간단하고 이해하기 쉽게 설명해 주세요."
        )
        # 난이도 기반 모델 사용 (토큰 스트리밍 허용)
        explanation = call_model_by_difficulty(
            state,
            system_prompt,
            user_prompt,
            tags=[],  # 스트리밍 허용
        ).strip()
        explain_result.explanation = explanation or explain_result.explanation

        # AIMessage를 messages에 추가하여 즉시 스트리밍 및 상태 저장
        if explanation:
            from langchain_core.messages import AIMessage

            ai_msg = AIMessage(content=explanation)
            state["messages"] = (state.get("messages") or []) + [ai_msg]
    else:
        explain_result.explanation = (
            explain_result.explanation or "설명할 대상을 찾을 수 없습니다."
        )

    state["explain_result"] = explain_result

    # 최종 응답에서 쉽게 사용할 수 있도록 final_output에도 넣어 둔다.
    final_output = state.setdefault("final_output", {})
    final_output["explain"] = {
        "explanation": explain_result.explanation,
        "examples": explain_result.examples,
    }
    state["final_output"] = final_output

    return state


builder = StateGraph(AgentState)
builder.add_node("Explain", explain)
builder.set_entry_point("Explain")
builder.add_edge("Explain", END)

graph = builder.compile()
