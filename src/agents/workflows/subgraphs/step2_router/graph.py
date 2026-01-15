"""Router subgraph.

Intent → IntentRoute/Planner/Executor/StepRouter 흐름의 뼈대입니다.
"""

from langgraph.graph import StateGraph

from agents.state import AgentState


def intent(state: AgentState) -> AgentState:
    """의도 감지 (stub)."""

    return state


def intent_route(state: AgentState) -> AgentState:
    """단일/복합 의도 분기 (stub)."""

    return state


def planner(state: AgentState) -> AgentState:
    """실행 계획 수립 (stub)."""

    return state


def executor(state: AgentState) -> AgentState:
    """단계 실행기 (stub)."""

    return state


def step_router(state: AgentState) -> AgentState:
    """Feature 라우팅 (stub)."""

    return state


builder = StateGraph(AgentState)
builder.add_node("Intent", intent)
builder.add_node("IntentRoute", intent_route)
builder.add_node("Planner", planner)
builder.add_node("Executor", executor)
builder.add_node("StepRouter", step_router)

builder.set_entry_point("Intent")
builder.add_edge("Intent", "IntentRoute")
builder.add_edge("IntentRoute", "Planner")
builder.add_edge("Planner", "Executor")
builder.add_edge("Executor", "StepRouter")

graph = builder.compile()
