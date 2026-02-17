# SSE Protocol v2

## 목적
`/stream`(v1)의 `data.type` 중심 스트림을 유지하면서, `/stream/v2`에서 SSE 표준 `event` 기반으로 이벤트 의미를 명확히 분리합니다.

- v1: 하위호환 유지
- v2: `id/event/data` + 공통 envelope + terminal event 표준화

## 구현 기준
현재 v2 구현은 LangGraph `astream_events(..., version="v2")` 기반입니다.

- 내부 런타임 이벤트(`on_chain_*`, `on_chat_model_*`, `on_tool_*`)를
  서비스 이벤트 taxonomy로 매핑해 전송합니다.
- v2에서는 `[DONE]` 문자열을 사용하지 않습니다.

## 엔드포인트
- `POST /stream` (v1, 유지)
- `POST /{agent_id}/stream` (v1, 유지)
- `POST /stream/v2` (v2)
- `POST /{agent_id}/stream/v2` (v2)

요청 바디는 `StreamInput`과 동일합니다.

예시:

```json
{
  "message": "안녕하세요",
  "streamTokens": true,
  "threadId": "test-thread-001",
  "userId": "test-user-001"
}
```

## SSE Frame 형식

```text
id: <run_id>:<seq>
event: <event_name>
data: <json>
```

- `id`: `<run_id>:<seq>`
- `event`: taxonomy 이름
- `data`: 공통 envelope + 이벤트 payload

## 공통 Envelope
모든 `data` JSON은 아래 공통 필드를 포함합니다.

```json
{
  "v": "2.0",
  "ts": "2026-02-17T14:30:00.000Z",
  "seq": 1,
  "run_id": "uuid",
  "thread_id": "uuid"
}
```

## 이벤트 taxonomy (현재 구현)

### 세션/런
- `session.metadata`
  - payload: `agent_id`, `capabilities`
- `run.started`
  - payload: `stream_tokens`
- `run.completed`
  - payload: `duration_ms`, `final_message_id?`
- `run.failed`
  - payload: `code`, `message`, `retryable`

terminal은 `run.completed` 또는 `run.failed` 중 정확히 1회입니다.

### 노드
- `node.started`
  - payload: `node`, `node_path`
- `node.progress`
  - payload: `node`, `message`
- `node.completed`
  - payload: `node`, `status`, `duration_ms`

매핑 기준:
- `on_chain_start` -> `node.started` / `node.progress`
- `on_chain_end` -> `node.completed`

### LLM
- `llm.message.started`
  - payload: `message_id`, `node`, `role`
- `llm.token.delta`
  - payload: `message_id`, `node`, `delta`, `index`
- `llm.message.completed`
  - payload: `message_id`, `finish_reason`

매핑 기준:
- `on_chat_model_start` -> `llm.message.started`
- `on_chat_model_stream` -> `llm.token.delta`
- `on_chat_model_end` -> `llm.message.completed`

### 메시지/툴/부가 이벤트
- `chat.message`
  - payload: `message_id`, `role`, `kind`, `content`, `node?`
- `tool.call.started`
  - payload: `tool_call_id`, `tool_name`
- `tool.call.completed`
  - payload: `tool_call_id`, `status`, `duration_ms`
- `credit.updated`
  - payload: `balance`, `total_cost`, `remaining`
- `artifact.ready`
  - payload: `artifact_id`, `name`, `mime`, `path`, `size`
- `heartbeat`
  - payload: `alive`

`artifact.ready`는 현재 `final_output.solution.pdf_path`가 관측될 때 emit합니다.

## `chat.message.kind` 표준값
- `status`
- `assistant_partial`
- `assistant_final`
- `tool_result`
- `review`
- `suggestion`
- `system_notice`

서버 매핑 요약:
- `ai + node=FinalResponse` -> `assistant_final`
- 일반 `ai` -> `assistant_partial`
- `tool` -> `tool_result`
- `custom + status 포함` -> `status`
- `custom + node=Review` -> `review`
- `custom + node=Suggestion` -> `suggestion`
- 그 외 -> `system_notice`

## 토큰 스트리밍 규칙
- `stream_tokens=true`: `llm.token.delta` 전송
- `stream_tokens=false`: `llm.token.delta` 미전송, 완결 `chat.message` 중심 전송

## 순서 보장/주의사항
- `seq`는 서버 전송 순서를 단조 증가로 보장합니다.
- `astream_events` 기반이라 `on_chain_start`가 먼저 관측되면 `node.started`가 토큰보다 먼저 옵니다.
- 다만 병렬 실행 노드가 있으면 서로 다른 노드 이벤트는 interleave될 수 있습니다.
  클라이언트는 `event + node + message_id + seq` 기준으로 처리해야 합니다.

## message_id 규칙
- `m_llm_<n>`: LLM 토큰 스트림 단위 식별자
- `m_chat_<n>`: 완결 `chat.message` 식별자
- `m_interrupt_<n>`, `m_error_<n>`: 인터럽트/에러 메시지 식별자

`message_id`는 run 범위 내 correlation id입니다.

## 예시
연결 직후:

```text
id: 8d1f...:1
event: session.metadata
data: {"v":"2.0","ts":"2026-02-17T14:30:00.000Z","seq":1,"run_id":"8d1f...","thread_id":"a21c...","agent_id":"tutor","capabilities":{"token_stream":true,"heartbeat":true,"terminal_event":true}}
```

토큰:

```text
id: 8d1f...:12
event: llm.token.delta
data: {"v":"2.0","ts":"2026-02-17T14:30:01.120Z","seq":12,"run_id":"8d1f...","thread_id":"a21c...","message_id":"m_llm_2","node":"FinalResponse","delta":"안","index":0}
```

종료:

```text
id: 8d1f...:45
event: run.completed
data: {"v":"2.0","ts":"2026-02-17T14:30:08.234Z","seq":45,"run_id":"8d1f...","thread_id":"a21c...","duration_ms":8234,"final_message_id":"m_chat_3"}
```

## 마이그레이션 가이드
1. 프론트는 `/stream/v2`로 opt-in 전환합니다.
2. `event` 기준 핸들러로 분기하고, `seq`를 기준으로 UI 상태를 반영합니다.
3. 운영 안정화 후 v1 deprecate 공지를 진행합니다.
4. 최종적으로 v1 제거를 검토합니다.
