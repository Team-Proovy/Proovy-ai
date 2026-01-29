"""\
CLI에서 /stream 엔드포인트를 호출해 SSE 응답을 관찰하고,
토큰 스트림을 하나의 최종 응답 텍스트로 재구성하는 유틸리티.

예시:
    uv run src/run_stream_client.py "2차 함수 개념 설명해줘"
    uv run src/run_stream_client.py --base-url http://localhost:8081 \
        --agent-id default "테스트"

서버는 src/run_service.py 로 띄워져 있어야 한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI /stream SSE 응답을 테스트하는 간단한 클라이언트",
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="에이전트에 보낼 사용자 메시지 (미입력 시 프롬프트로 입력)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8081",
        help="서비스의 기본 URL (기본: http://127.0.0.1:8081)",
    )
    parser.add_argument(
        "--agent-id",
        default=None,
        help="타겟 에이전트 ID (예: research-assistant). 미설정 시 /stream 사용",
    )
    parser.add_argument(
        "--no-stream-tokens",
        dest="stream_tokens",
        action="store_false",
        help="토큰 스트리밍을 끄고 message 이벤트만 확인",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="받은 모든 SSE data payload(JSON)를 순서대로 출력",
    )
    return parser


async def _run_stream_client(args: argparse.Namespace) -> None:
    # 1) 메시지 확보
    message = args.message or input("입력할 메시지: ")

    # 2) 요청 바디(StreamInput 스키마와 맞춤)
    payload: dict[str, Any] = {
        "message": message,
        "model": None,
        "thread_id": None,
        "user_id": None,
        "agent_config": {},
        "files_url": None,
        "chosen_features": ["Solve"],
        "stream_tokens": bool(args.stream_tokens),
    }

    base_url = args.base_url.rstrip("/")
    if args.agent_id:
        url = f"{base_url}/{args.agent_id}/stream"
    else:
        url = f"{base_url}/stream"

    print(f"\n[요청] POST {url}")
    print(f"payload: {json.dumps(payload, ensure_ascii=False)}\n")

    # 3) SSE 스트림 열기
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()

            full_token_text = ""
            events: list[dict[str, Any]] = []
            message_events: list[dict[str, Any]] = []

            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    # SSE의 다른 필드(id:, event: 등)는 지금은 무시
                    continue

                data_str = line[len("data: ") :].strip()

                if data_str == "[DONE]":
                    # 서버에서 스트림 종료를 알림
                    break

                try:
                    payload_obj = json.loads(data_str)
                except json.JSONDecodeError:
                    print(f"[경고] JSON 파싱 불가 data: {data_str}")
                    continue

                if isinstance(payload_obj, dict):
                    events.append(payload_obj)
                else:
                    # dict 가 아닌 경우는 디버깅용으로만 출력
                    events.append({"_raw": payload_obj})

                if args.show_events:
                    print("[event]", json.dumps(payload_obj, ensure_ascii=False))

                if not isinstance(payload_obj, dict):
                    continue

                event_type = payload_obj.get("type")

                if event_type == "token":
                    # 토큰 스트림: content 문자열을 순서대로 이어붙인다.
                    token_text = payload_obj.get("content") or ""
                    full_token_text += token_text

                elif event_type == "message":
                    # ChatMessage 스키마(dict)를 그대로 보관
                    message_events.append(payload_obj.get("content") or {})

                elif event_type == "error":
                    print("[서버 에러]", payload_obj.get("content"))

    # 4) 요약 출력
    print("\n===== 토큰 스트림으로 재구성한 최종 응답 =====\n")
    if full_token_text:
        print(full_token_text)
    else:
        print("(토큰 스트림이 없거나 비어 있습니다)")

    if message_events:
        print("\n===== 마지막 message 이벤트 (ChatMessage) =====\n")
        last_msg = message_events[-1]
        # ChatMessage(content=..., type=..., tool_calls=...) 구조를 가정
        content = last_msg.get("content")
        msg_type = last_msg.get("type")
        print(f"type: {msg_type}")
        print("content:\n")
        print(content)

    if not full_token_text and not message_events:
        print("\n(유효한 token/message 이벤트를 받지 못했습니다)")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run_stream_client(args))


if __name__ == "__main__":
    main()
