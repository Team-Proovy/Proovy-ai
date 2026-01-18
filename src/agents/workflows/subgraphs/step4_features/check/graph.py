"""Check feature subgraph."""

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def check(state: AgentState) -> AgentState:
    """풀이 검토 기능 (stub)."""
    print("---FEATURE: CHECK---")
    # TODO: Implement actual logic for check
    state["prev_action"] = "Check"
    state["next_action"] = "Executor"
    return state


builder = StateGraph(AgentState)
builder.add_node("Check", check)
builder.set_entry_point("Check")
builder.add_edge("Check", END)

graph = builder.compile()
