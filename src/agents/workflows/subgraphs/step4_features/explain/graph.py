"""Explain feature subgraph.

단일 노드에서 사용자의 마지막 질문을 간단히 풀어 설명하는 용도이다.
과한 모델/파이프라인을 쓰지 않고, 가벼운 LLM 한 번만 호출한다.
"""

from langgraph.graph import END, StateGraph

from agents.state import AgentState, ExplainResult
from agents.workflows.utils import call_model, recent_user_context
from schema.models import OpenRouterModelName


def explain(state: AgentState) -> AgentState:
    print("---FEATURE: EXPLAIN---")

    # 최근 사용자 메시지를 가져온다.
    user_text = recent_user_context(state, max_messages=1)
    explain_result = state.get("explain_result") or ExplainResult()

    if user_text:
        system_prompt = (
            "You are a kind Korean tutor. "
            "Explain the given concept or question in very simple Korean, "
            "using short sentences and, if helpful, 1-2 easy examples."
        )
        user_prompt = f"사용자 질문 또는 개념:\n{user_text}\n\n간단하고 이해하기 쉽게 설명해 주세요."
        explanation = call_model(
            OpenRouterModelName.GPT_5_MINI,
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
