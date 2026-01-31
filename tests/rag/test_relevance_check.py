from __future__ import annotations

from agents.workflows.subgraphs.step3_rag.graph import relevance_check


def test_relevance_check_pass_with_docs() -> None:
    state = {
        "tool_outputs": {
            "retrieved_docs": [
                {"id": "doc1", "title": "t", "text": "x", "score": 0.9, "metadata": {}}
            ]
        }
    }
    relevance_check(state)
    assert state["tool_outputs"]["rag_relevance_passed"] is True
    assert state["prev_action"] == "RAG"
    assert state["next_action"] == "IntentRoute"


def test_relevance_check_fail_when_empty() -> None:
    state = {"tool_outputs": {"retrieved_docs": []}}
    relevance_check(state)
    assert state["tool_outputs"]["rag_relevance_passed"] is False


def test_relevance_check_normalizes_doc() -> None:
    state = {"tool_outputs": {"retrieved_docs": [{"content": "hi", "score": "0.7"}]}}
    relevance_check(state)
    normalized = state["tool_outputs"]["retrieved_docs"][0]
    assert normalized["text"] == "hi"
    assert isinstance(normalized["score"], float)

