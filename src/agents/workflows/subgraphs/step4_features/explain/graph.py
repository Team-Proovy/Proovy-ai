"""Explain feature subgraph.

단일 노드에서 사용자의 마지막 질문을 간단히 풀어 설명하는 용도이다.
<<<<<<< HEAD
난이도에 따라 적절한 LLM 모델을 선택하여 호출한다.

난이도별 모델:
- easy: Gemini 2.5 Flash
- medium: Gemini 3 Flash
- hard: Gemini 3 Pro
=======
과한 모델/파이프라인을 쓰지 않고, 가벼운 LLM 한 번만 호출한다.

checkpointer가 저장한 대화 히스토리를 활용하여 멀티턴 대화를 지원한다.
>>>>>>> cf7bf04949b03632090aaef8e08b8bf24ef21ffa
"""

from langgraph.graph import END, StateGraph

from agents.state import AgentState, ExplainResult
from agents.workflows.utils import (
<<<<<<< HEAD
    call_model_by_difficulty,
    get_difficulty_from_state,
    recent_user_context,
)
=======
    call_model,
    get_conversation_summary,
    recent_user_context,
)
from schema.models import OpenRouterModelName
>>>>>>> cf7bf04949b03632090aaef8e08b8bf24ef21ffa


def explain(state: AgentState) -> AgentState:
    print("---FEATURE: EXPLAIN---")

    # 최근 사용자 메시지를 가져온다 (대화 맥락 포함)
    user_text = recent_user_context(state, max_messages=3, include_assistant=True)
    explain_result = state.get("explain_result") or ExplainResult()

<<<<<<< HEAD
    difficulty = get_difficulty_from_state(state)
    print(f"→ Explaining with difficulty: {difficulty}")
=======
    # 이전 대화 맥락 수집 (멀티턴 지원)
    conversation_context = get_conversation_summary(state, max_chars=1000)
>>>>>>> cf7bf04949b03632090aaef8e08b8bf24ef21ffa

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
<<<<<<< HEAD
        user_prompt = f"사용자 질문 또는 개념:\n{user_text}\n\n간단하고 이해하기 쉽게 설명해 주세요."
        # 난이도 기반 모델 사용
        explanation = call_model_by_difficulty(
            state,
=======
        explanation = call_model(
            OpenRouterModelName.GPT_5_MINI,
>>>>>>> cf7bf04949b03632090aaef8e08b8bf24ef21ffa
            system_prompt,
            user_prompt,
        ).strip()
        explain_result.explanation = explanation or explain_result.explanation
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
