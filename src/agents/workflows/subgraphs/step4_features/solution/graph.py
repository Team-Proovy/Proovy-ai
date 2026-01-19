"""Solution feature subgraph."""

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def solution(state: AgentState) -> AgentState:
    """해설 기능 (stub)."""
    print("---FEATURE: SOLUTION---")
    # TODO: Implement actual logic for solution
    state["prev_action"] = "Solution"
    state["next_action"] = "Executor"
    return state


builder = StateGraph(AgentState)
builder.add_node("Solution", solution)
builder.set_entry_point("Solution")
builder.add_edge("Solution", END)

graph = builder.compile()
