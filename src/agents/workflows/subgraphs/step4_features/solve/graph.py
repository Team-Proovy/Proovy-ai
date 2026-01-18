"""Solve feature subgraph."""

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def solve(state: AgentState) -> AgentState:
    """문제 풀이 기능 (stub)."""
    print("---FEATURE: SOLVE---")
    # TODO: Implement actual logic for solve
    state["prev_action"] = "Solve"
    state["next_action"] = "Executor"
    return state


builder = StateGraph(AgentState)
builder.add_node("Solve", solve)
builder.set_entry_point("Solve")
builder.add_edge("Solve", END)

graph = builder.compile()
