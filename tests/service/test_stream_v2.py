from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from schema import StreamInput
from service import service as service_module


class _FakeAgent:
    def __init__(
        self,
        events_v1: list[Any] | None = None,
        events_v2: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        delay_sec: float = 0.0,
    ) -> None:
        self._events_v1 = events_v1 or []
        self._events_v2 = events_v2 or []
        self._error = error
        self._delay_sec = delay_sec

    async def astream(self, **kwargs: Any):  # noqa: ANN003
        for event in self._events_v1:
            await asyncio.sleep(self._delay_sec)
            yield event
        if self._error is not None:
            raise self._error

    async def astream_events(self, *args: Any, **kwargs: Any):  # noqa: ANN003
        for event in self._events_v2:
            await asyncio.sleep(self._delay_sec)
            yield event
        if self._error is not None:
            raise self._error


async def _fake_handle_input(
    user_input: StreamInput,  # noqa: ARG001
    agent: Any,  # noqa: ANN401, ARG001
) -> tuple[dict[str, Any], uuid.UUID, str]:
    return {"input": {}, "config": {}}, uuid.UUID(
        "00000000-0000-0000-0000-000000000111"
    ), "thread-test"


async def _collect_chunks(generator) -> list[str]:  # noqa: ANN001
    chunks: list[str] = []
    async for chunk in generator:
        chunks.append(chunk)
    return chunks


def _patch_stream_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events_v1: list[Any] | None = None,
    events_v2: list[dict[str, Any]] | None = None,
    error: Exception | None = None,
    delay_sec: float = 0.0,
) -> None:
    fake_agent = _FakeAgent(
        events_v1=events_v1,
        events_v2=events_v2,
        error=error,
        delay_sec=delay_sec,
    )
    monkeypatch.setattr(service_module, "get_agent", lambda _agent_id: fake_agent)
    monkeypatch.setattr(service_module, "_handle_input", _fake_handle_input)


def _parse_v2_chunks(chunks: list[str]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for chunk in chunks:
        frame: dict[str, Any] = {}
        for line in chunk.strip().splitlines():
            if line.startswith("id:"):
                frame["id"] = line[len("id:") :].strip()
            elif line.startswith("event:"):
                frame["event"] = line[len("event:") :].strip()
            elif line.startswith("data:"):
                frame["data"] = json.loads(line[len("data:") :].strip())
        if frame:
            parsed.append(frame)
    return parsed


def _parse_v1_data(chunk: str) -> str:
    assert chunk.startswith("data: ")
    return chunk[len("data: ") :].strip()


@pytest.mark.asyncio
async def test_stream_v2_event_order_and_terminal_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_v2 = [
        {
            "event": "on_chain_start",
            "name": "Intent",
            "metadata": {"langgraph_node": "Intent", "langgraph_path": "Main/Intent"},
            "data": {"input": {}},
        },
        {
            "event": "on_chain_end",
            "name": "Intent",
            "metadata": {"langgraph_node": "Intent", "langgraph_path": "Main/Intent"},
            "data": {
                "output": {
                    "credit_state": {"balance": 100, "total_cost": 1, "difficulty": "easy"}
                }
            },
        },
        {
            "event": "on_chain_start",
            "name": "FinalResponse",
            "metadata": {
                "langgraph_node": "FinalResponse",
                "langgraph_path": "Main/FinalResponse",
            },
            "data": {"input": {}},
        },
        {
            "event": "on_chat_model_start",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "FinalResponse"},
            "tags": [],
            "data": {"input": {"messages": []}},
        },
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "FinalResponse"},
            "tags": [],
            "data": {"chunk": AIMessageChunk(content="안")},
        },
        {
            "event": "on_chat_model_end",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "FinalResponse"},
            "tags": [],
            "data": {
                "output": AIMessage(
                    content="안",
                    response_metadata={"finish_reason": "stop"},
                )
            },
        },
        {
            "event": "on_chain_end",
            "name": "FinalResponse",
            "metadata": {
                "langgraph_node": "FinalResponse",
                "langgraph_path": "Main/FinalResponse",
            },
            "data": {"output": {"messages": [AIMessage(content="안녕하세요")]}},
        },
    ]
    _patch_stream_dependencies(monkeypatch, events_v2=events_v2)

    user_input = StreamInput(message="테스트", stream_tokens=True)
    chunks = await _collect_chunks(service_module.message_generator_v2(user_input))
    parsed = _parse_v2_chunks(chunks)
    names = [frame["event"] for frame in parsed]

    assert names[0] == "session.metadata"
    assert names[1] == "run.started"
    assert "node.started" in names
    assert "node.progress" in names
    assert "llm.message.started" in names
    assert "llm.token.delta" in names
    assert "llm.message.completed" in names
    assert names[-1] == "run.completed"

    terminals = [name for name in names if name in {"run.completed", "run.failed"}]
    assert terminals == ["run.completed"]

    seqs = [int(frame["data"]["seq"]) for frame in parsed]
    assert seqs == sorted(seqs)

    final_node_started_seq = next(
        frame["data"]["seq"]
        for frame in parsed
        if frame["event"] == "node.started"
        and frame["data"].get("node") == "FinalResponse"
    )
    first_token_seq = next(
        frame["data"]["seq"]
        for frame in parsed
        if frame["event"] == "llm.token.delta"
    )
    assert final_node_started_seq < first_token_seq


@pytest.mark.asyncio
async def test_stream_v2_disables_llm_token_delta_when_stream_tokens_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_v2 = [
        {
            "event": "on_chain_start",
            "name": "FinalResponse",
            "metadata": {
                "langgraph_node": "FinalResponse",
                "langgraph_path": "Main/FinalResponse",
            },
            "data": {"input": {}},
        },
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "FinalResponse"},
            "tags": [],
            "data": {"chunk": AIMessageChunk(content="안")},
        },
        {
            "event": "on_chain_end",
            "name": "FinalResponse",
            "metadata": {
                "langgraph_node": "FinalResponse",
                "langgraph_path": "Main/FinalResponse",
            },
            "data": {"output": {"messages": [AIMessage(content="최종 답변")]}},
        },
    ]
    _patch_stream_dependencies(monkeypatch, events_v2=events_v2)

    user_input = StreamInput(message="테스트", stream_tokens=False)
    chunks = await _collect_chunks(service_module.message_generator_v2(user_input))
    parsed = _parse_v2_chunks(chunks)
    names = [frame["event"] for frame in parsed]

    assert "llm.token.delta" not in names
    assert "chat.message" in names

    chat_payloads = [frame["data"] for frame in parsed if frame["event"] == "chat.message"]
    assert any(payload.get("kind") == "assistant_final" for payload in chat_payloads)


@pytest.mark.asyncio
async def test_stream_v2_emits_heartbeat_on_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_v2 = [
        {
            "event": "on_chain_start",
            "name": "Intent",
            "metadata": {"langgraph_node": "Intent", "langgraph_path": "Main/Intent"},
            "data": {"input": {}},
        },
    ]
    monkeypatch.setattr(service_module, "STREAM_V2_HEARTBEAT_INTERVAL_SEC", 0.01)
    _patch_stream_dependencies(monkeypatch, events_v2=events_v2, delay_sec=0.03)

    user_input = StreamInput(message="테스트", stream_tokens=True)
    chunks = await _collect_chunks(service_module.message_generator_v2(user_input))
    parsed = _parse_v2_chunks(chunks)
    names = [frame["event"] for frame in parsed]

    assert "heartbeat" in names


@pytest.mark.asyncio
async def test_stream_v2_emits_single_run_failed_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_v2 = [
        {
            "event": "on_chain_start",
            "name": "Intent",
            "metadata": {"langgraph_node": "Intent", "langgraph_path": "Main/Intent"},
            "data": {"input": {}},
        },
    ]
    _patch_stream_dependencies(
        monkeypatch,
        events_v2=events_v2,
        error=RuntimeError("boom"),
    )

    user_input = StreamInput(message="테스트", stream_tokens=True)
    chunks = await _collect_chunks(service_module.message_generator_v2(user_input))
    parsed = _parse_v2_chunks(chunks)
    names = [frame["event"] for frame in parsed]

    terminals = [name for name in names if name in {"run.completed", "run.failed"}]
    assert terminals == ["run.failed"]

    failed_payload = next(
        frame["data"] for frame in parsed if frame["event"] == "run.failed"
    )
    assert failed_payload["code"] == "internal_error"
    assert failed_payload["retryable"] is False


@pytest.mark.asyncio
async def test_stream_v1_backward_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    events_v1 = [
        (
            "FinalResponse",
            "messages",
            (
                AIMessageChunk(content="안"),
                {"tags": []},
            ),
        ),
    ]
    _patch_stream_dependencies(monkeypatch, events_v1=events_v1)

    user_input = StreamInput(message="테스트", stream_tokens=True)
    chunks = await _collect_chunks(service_module.message_generator(user_input))

    assert chunks[-1] == "data: [DONE]\n\n"

    payloads: list[dict[str, Any]] = []
    for chunk in chunks[:-1]:
        data = _parse_v1_data(chunk)
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))

    assert payloads[0]["type"] == "thread_id"
    assert payloads[0]["thread_id"] == "thread-test"
    assert any(payload.get("type") == "token" for payload in payloads)


def test_stream_v2_routes_registered() -> None:
    paths = {route.path for route in service_module.app.routes}
    assert "/stream/v2" in paths
    assert "/{agent_id}/stream/v2" in paths
