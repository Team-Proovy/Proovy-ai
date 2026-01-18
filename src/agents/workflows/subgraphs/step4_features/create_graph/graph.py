"""Create Graph feature subgraph."""

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def create_graph(state: AgentState) -> AgentState:
    """그래프 생성 기능 (stub)."""
    print("---FEATURE: CREATE_GRAPH---")
    # TODO: Implement actual logic for create_graph
    state["prev_action"] = "Create_graph"
    state["next_action"] = "Executor"
    return state


builder = StateGraph(AgentState)
builder.add_node("CreateGraph", create_graph)
builder.set_entry_point("CreateGraph")
builder.add_edge("CreateGraph", END)

graph = builder.compile()
