"""RAG subgraph.

EmbeddingSearch → RelevanceCheck → RetrievedDocs 흐름의 뼈대입니다.
"""

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def embedding_search(state: AgentState) -> AgentState:
    """유사도 검색 (stub)."""

    return state


def relevance_check(state: AgentState) -> AgentState:
    """유사도 threshold 체크 (stub)."""

    return state


def retrieved_docs(state: AgentState) -> AgentState:
    """검색 결과 주입 (stub)."""

    return state


builder = StateGraph(AgentState)
builder.add_node("EmbeddingSearch", embedding_search)
builder.add_node("RelevanceCheck", relevance_check)
builder.add_node("RetrievedDocs", retrieved_docs)

builder.set_entry_point("EmbeddingSearch")
builder.add_edge("EmbeddingSearch", "RelevanceCheck")
builder.add_edge("RelevanceCheck", "RetrievedDocs")
builder.add_edge("RetrievedDocs", END)

graph = builder.compile()
