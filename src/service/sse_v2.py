"""SSE v2 protocol helpers.

This module centralizes:
- envelope creation (v/ts/seq/run_id/thread_id)
- SSE frame formatting (id/event/data)
- chat.message role/kind mapping
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SSE_V2_VERSION = "2.0"

CHAT_MESSAGE_KIND_STATUS = "status"
CHAT_MESSAGE_KIND_ASSISTANT_PARTIAL = "assistant_partial"
CHAT_MESSAGE_KIND_ASSISTANT_FINAL = "assistant_final"
CHAT_MESSAGE_KIND_TOOL_RESULT = "tool_result"
CHAT_MESSAGE_KIND_REVIEW = "review"
CHAT_MESSAGE_KIND_SUGGESTION = "suggestion"
CHAT_MESSAGE_KIND_SYSTEM_NOTICE = "system_notice"


def utc_now_iso() -> str:
    """Return a UTC ISO8601 timestamp with milliseconds and trailing Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass
class SseV2Emitter:
    """Formats protocol-compliant SSE v2 frames with a shared envelope."""

    run_id: str
    thread_id: str
    seq: int = 0

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> str:
        if not event or "\n" in event:
            raise ValueError("event name must be a non-empty single line")
        self.seq += 1
        data: dict[str, Any] = {
            "v": SSE_V2_VERSION,
            "ts": utc_now_iso(),
            "seq": self.seq,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
        }
        if payload:
            data.update(payload)
        event_id = f"{self.run_id}:{self.seq}"
        return (
            f"id: {event_id}\n"
            f"event: {event}\n"
            f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        )


def chat_role_from_type(message_type: str) -> str:
    if message_type == "human":
        return "user"
    if message_type == "ai":
        return "assistant"
    if message_type == "tool":
        return "tool"
    return "system"


def chat_kind_from_message(
    *,
    message_type: str,
    node_name: str | None = None,
    custom_data: dict[str, Any] | None = None,
) -> str:
    node = (node_name or "").strip()
    custom = custom_data or {}

    if message_type == "tool":
        return CHAT_MESSAGE_KIND_TOOL_RESULT

    if message_type == "custom":
        if "status" in custom:
            return CHAT_MESSAGE_KIND_STATUS
        if node == "Review":
            return CHAT_MESSAGE_KIND_REVIEW
        if node == "Suggestion":
            return CHAT_MESSAGE_KIND_SUGGESTION
        return CHAT_MESSAGE_KIND_SYSTEM_NOTICE

    if message_type == "ai":
        if node == "FinalResponse":
            return CHAT_MESSAGE_KIND_ASSISTANT_FINAL
        if node == "Review":
            return CHAT_MESSAGE_KIND_REVIEW
        if node == "Suggestion":
            return CHAT_MESSAGE_KIND_SUGGESTION
        return CHAT_MESSAGE_KIND_ASSISTANT_PARTIAL

    return CHAT_MESSAGE_KIND_SYSTEM_NOTICE

