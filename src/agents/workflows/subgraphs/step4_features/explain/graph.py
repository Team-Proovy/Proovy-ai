"""Explain feature subgraph."""

from langgraph.graph import END, StateGraph

from agents.state import AgentState, ExplainResult


def explain(state: AgentState) -> AgentState:
    print("---FEATURE: EXPLAIN---")
    explain_result = state.get("explain_result") or ExplainResult()
    explain_result.explanation = (
        explain_result.explanation or "간단한 용어 설명이 준비되지 않았습니다."
    )
    state["explain_result"] = explain_result
    state["prev_action"] = "Explain"
    state["next_action"] = "Executor"
    return state


builder = StateGraph(AgentState)
builder.add_node("Explain", explain)
builder.set_entry_point("Explain")
builder.add_edge("Explain", END)

graph = builder.compile()
