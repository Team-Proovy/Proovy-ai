from langchain_core.messages import AIMessage, HumanMessage

from agents.workflows.utils import (
    get_conversation_history,
    get_conversation_summary,
    references_previous_conversation,
)


def _state_with_messages(messages):
    return {"messages": messages}


def test_references_previous_conversation_avoids_generic_false_positive():
    assert not references_previous_conversation("이전 단원에서 배운 공식 알려줘")
    assert not references_previous_conversation("전에 이런 유형 본 적 있어?")
    assert references_previous_conversation("이전 대화에서 푼 문제 다시 설명해줘")
    assert references_previous_conversation("전에 풀었던 문제 다시 보여줘")


def test_get_conversation_summary_excludes_current_user_and_keeps_user_first():
    messages = [
        HumanMessage(content="h1"),
        AIMessage(content="a1"),
        HumanMessage(content="h2"),
        AIMessage(content="a2"),
        HumanMessage(content="current"),
    ]
    state = _state_with_messages(messages)

    summary = get_conversation_summary(state, max_chars=2000, max_turns=2)
    lines = summary.splitlines()

    assert lines
    assert lines[0].startswith("사용자:")
    assert "current" not in summary
    assert "h1" in summary
    assert "a2" in summary


def test_get_conversation_history_does_not_start_with_orphan_ai():
    messages = [
        HumanMessage(content="h1"),
        AIMessage(content="a1"),
        HumanMessage(content="h2"),
        AIMessage(content="a2"),
        HumanMessage(content="h3"),
        AIMessage(content="a3"),
        HumanMessage(content="h4"),
    ]
    state = _state_with_messages(messages)

    history = get_conversation_history(state, max_turns=2, max_chars_per_message=2000)

    assert [message.content for message in history] == ["h2", "a2", "h3", "a3", "h4"]
    assert history[0].type in {"human", "user"}
