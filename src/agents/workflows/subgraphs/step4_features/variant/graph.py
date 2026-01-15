"""Variant feature subgraph."""

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def variant(state: AgentState) -> AgentState:
    """유사 문제 생성 기능 (stub)."""
    print("---FEATURE: VARIANT---")
    # TODO: Implement actual logic for variant
    return state


builder = StateGraph(AgentState)
builder.add_node("Variant", variant)
builder.set_entry_point("Variant")
builder.add_edge("Variant", END)

graph = builder.compile()
