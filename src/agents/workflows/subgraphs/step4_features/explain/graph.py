"""Explain feature subgraph."""

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def explain(state: AgentState) -> AgentState:
    """개념 설명 기능 (stub)."""
    print("---FEATURE: EXPLAIN---")
    # TODO: Implement actual logic for explain
    return state


builder = StateGraph(AgentState)
builder.add_node("Explain", explain)
builder.set_entry_point("Explain")
builder.add_edge("Explain", END)

graph = builder.compile()
