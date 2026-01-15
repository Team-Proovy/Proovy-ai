"""Features subgraph.

Solve / Explain / Graph / Variant / Solution / Check 노드들을
하나의 서브그래프로 묶는 뼈대 구현입니다.
"""

from langgraph.graph import StateGraph

from agents.state import AgentState


def feature_solve(state: AgentState) -> AgentState:
    return state


def feature_explain(state: AgentState) -> AgentState:
    return state


def feature_graph(state: AgentState) -> AgentState:
    return state


def feature_variant(state: AgentState) -> AgentState:
    return state


def feature_solution(state: AgentState) -> AgentState:
    return state


def feature_check(state: AgentState) -> AgentState:
    return state


builder = StateGraph(AgentState)
builder.add_node("Solve", feature_solve)
builder.add_node("Explain", feature_explain)
builder.add_node("Graph", feature_graph)
builder.add_node("Variant", feature_variant)
builder.add_node("Solution", feature_solution)
builder.add_node("Check", feature_check)

builder.set_entry_point("Solve")
builder.add_edge("Solve", "Explain")
builder.add_edge("Explain", "Graph")
builder.add_edge("Graph", "Variant")
builder.add_edge("Variant", "Solution")
builder.add_edge("Solution", "Check")

graph = builder.compile()
