"""Main LangGraph definition for the Proovy agent.

상위 레벨에서만 Preprocessing / Router / RAG / Features / Review
같은 큰 노드들만 보이도록 구성한 메인 그래프입니다.
각 노드는 실제 로직을 담당하는 서브그래프를 호출합니다.
"""

from langgraph.graph import StateGraph

from agents.state import AgentState
from agents.workflows.subgraphs.step1_preprocessing.graph import (
    graph as preprocessing_graph,
)
from agents.workflows.subgraphs.step2_router.graph import graph as router_graph
from agents.workflows.subgraphs.step3_rag.graph import graph as rag_graph
from agents.workflows.subgraphs.step4_features import graph as features_graph


def review(state: AgentState) -> AgentState:
    """Review/후처리 레이어 (stub)."""

    return state


def final_response(state: AgentState) -> AgentState:
    """최종 Response 노드 (stub)."""

    return state


builder = StateGraph(AgentState)

# 상위 레벨 노드들은 서브그래프들을 그대로 노드로 사용
builder.add_node("Preprocessing", preprocessing_graph)
builder.add_node("Router", router_graph)
builder.add_node("RAG", rag_graph)
builder.add_node("Features", features_graph)
builder.add_node("Review", review)
builder.add_node("Response", final_response)

builder.set_entry_point("Preprocessing")
builder.add_edge("Preprocessing", "Router")
builder.add_edge("Router", "RAG")
builder.add_edge("RAG", "Features")
builder.add_edge("Features", "Review")
builder.add_edge("Review", "Response")

# This is the object referenced by langgraph.json:
# "agent": "src.agents.workflows.maingraph:graph"
graph = builder.compile()
