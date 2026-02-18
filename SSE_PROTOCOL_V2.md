# SSE Protocol v2

## 목적
`/stream`(v1)의 `data.type` 중심 스트림을 유지하면서, `/stream/v2`에서 SSE 표준 `event` 기반으로 이벤트 의미를 명확히 분리합니다.

- v1: 하위호환 유지
- v2: `id/event/data` + 공통 envelope + terminal event 표준화

## 구현 기준
현재 v2 구현은 LangGraph `astream_events(..., version="v2")` 기반입니다.

- 내부 런타임 이벤트(`on_chain_*`, `on_chat_model_*`, `on_tool_*`, `on_custom_event`)를
  서비스 이벤트 taxonomy로 매핑해 전송합니다.
- v2에서는 `[DONE]` 문자열을 사용하지 않습니다. terminal event는 `run.completed` 또는 `run.failed` 중 하나입니다.
- 모든 `data`는 UTF-8 JSON으로 전송되며(`ensure_ascii=False`), `v/ts/seq/run_id/thread_id` 키는 payload에서 무시됩니다.

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

- `seq`는 SSE 프레임 단위로 1씩 증가합니다.
- `id`는 `run_id:seq` 포맷이며 클라이언트 재연결을 위한 기준으로 사용할 수 있습니다.

## 이벤트 taxonomy (현재 구현)

## 이벤트/데이터 형식 표

### 세션/런
| event | data 필드 | 설명 | 예시 payload |
| --- | --- | --- | --- |
| session.metadata | agent_id, capabilities | 연결 직후 1회 전송되는 세션 메타 | `{"agent_id":"tutor","capabilities":{"token_stream":true,"heartbeat":true,"terminal_event":true}}` |
| run.started | stream_tokens | 스트림 시작(토큰 스트리밍 여부 포함) | `{"stream_tokens":true}` |
| run.completed | duration_ms, final_message_id? | 정상 종료(terminal) | `{"duration_ms":8234,"final_message_id":"m_chat_3"}` |
| run.failed | code, message, retryable | 실패 종료(terminal) | `{"code":"internal_error","message":"Internal server error","retryable":false}` |

### 노드
| event | data 필드 | 설명 | 예시 payload |
| --- | --- | --- | --- |
| node.started | node, node_path | 노드 시작(visible 노드만) | `{"node":"FinalResponse","node_path":"Main/FinalResponse"}` |
| node.progress | node, message | 진행 메시지(1회) | `{"node":"Solve","message":"문제를 분석하고 필요한 정보를 정리하고 있습니다."}` |
| node.completed | node, status, duration_ms | 노드 종료(현재 status=success) | `{"node":"FinalResponse","status":"success","duration_ms":1200}` |

### LLM
| event | data 필드 | 설명 | 예시 payload |
| --- | --- | --- | --- |
| llm.message.started | message_id, node, role | LLM 메시지 시작 | `{"message_id":"m_llm_2","node":"FinalResponse","role":"assistant"}` |
| llm.token.delta | message_id, node, delta, index | 토큰 델타 스트림 | `{"message_id":"m_llm_2","node":"FinalResponse","delta":"안","index":0}` |
| llm.message.completed | message_id, finish_reason | LLM 메시지 종료(stop/switch/error) | `{"message_id":"m_llm_2","finish_reason":"stop"}` |

### 메시지/툴/부가
| event | data 필드 | 설명 | 예시 payload |
| --- | --- | --- | --- |
| chat.message | message_id, role, kind, content, node? | 완결 메시지/커스텀/인터럽트 | `{"message_id":"m_chat_3","role":"assistant","kind":"assistant_final","content":"안녕하세요","node":"FinalResponse"}` |
| tool.call.started | tool_call_id, tool_name | 툴 호출 시작 | `{"tool_call_id":"tool_call_1","tool_name":"calculator"}` |
| tool.call.completed | tool_call_id, status, duration_ms | 툴 호출 종료(현재 status=success) | `{"tool_call_id":"tool_call_1","status":"success","duration_ms":230}` |
| credit.updated | balance, total_cost, remaining | 크레딧 상태 변경 | `{"balance":100,"total_cost":1,"remaining":99}` |
| artifact.ready | artifact_id, name, mime, path, size | 결과물 준비 완료 | `{"artifact_id":"solution.pdf","name":"solution.pdf","mime":"application/pdf","path":"/tmp/solution.pdf","size":102400}` |
| heartbeat | alive | 유휴 상태 heartbeat | `{"alive":true}` |

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
`final_message_id`는 `assistant_final` 또는 `interrupt` 계열 메시지가 emit된 경우에만 포함됩니다.

런 시작/종료 흐름:
- 연결 직후 `session.metadata` -> `run.started` 순으로 항상 1회씩 전송됩니다.
- 에러 발생 시 `run.failed`가 전송되고, 정상 종료 시 `run.completed`가 전송됩니다.
- 현재 구현에서 `run.failed.code=internal_error`, `retryable=false`로 고정됩니다.

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

추가 규칙:
- `node.*`는 `VISIBLE_NODES`에 포함된 노드만 전송됩니다.
- `node_path`는 `langgraph_path`/`node_path` 메타데이터에서 생성하며 없으면 `node`로 대체합니다.
- LLM 토큰 이벤트가 먼저 들어오는 경우에도 `node.started`가 보정 전송될 수 있습니다.
- 현재 `node.completed.status`는 `success`로 고정됩니다.

`VISIBLE_NODES`:
`FinalResponse`, `Simple_response`, `Fallback`, `CreditInsufficient`, `CreditCheck`,
`Router`, `Preprocessing`, `RAG`, `Solve`, `Explain`, `CreateGraph`, `Variant`,
`Solution`, `Check` 및 아래 `PROGRESS_MESSAGES` 목록

`PROGRESS_MESSAGES` (node.progress 메시지 텍스트):
`CheckType`, `FileConvert`, `VisionLLM`, `Intent`, `Planner`, `Executor`, `RetryCounter`,
`EmbeddingSearch`, `RelevanceCheck`, `RetrievedDocs`, `Solve_Analysis`,
`Solve_Strategy`, `Solve_Computation`, `Solve_Writer`, `Explain`, `Explain_Writer`,
`CreateGraph`, `Variant`, `Solution`, `Check`, `Review`, `Suggestion`

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

추가 규칙:
- `stream_tokens=false`면 `llm.*` 이벤트는 전송하지 않습니다.
- `tags`에 `skip_stream`이 있으면 `llm.*` 이벤트는 전송하지 않습니다.
- `on_chat_model_stream`이 먼저 도착하면 내부적으로 `llm.message.started`를 선행 발행합니다.
- 새로운 LLM 메시지가 시작되는데 이전 LLM 메시지가 종료되지 않았다면
  `llm.message.completed`(`finish_reason: "switch"`)가 먼저 전송됩니다.

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

추가 규칙:
- `chat.message`는 동일 메시지 중복 전송을 막기 위해 `(type, kind, node, content_hash)` 기준으로 dedupe됩니다.
- 사용자 입력과 동일한 `human` 메시지는 전송하지 않습니다.
- `on_custom_event`는 `custom` 메시지로 변환하여 `chat.message`로 전송합니다.
- `__interrupt__`가 state에서 관측되면 `chat.message(kind=system_notice)`로 전송됩니다.
- `tool_call_id`는 `stream_event.run_id`가 있으면 사용하고, 없으면 내부 생성 id를 사용합니다.
- 현재 `tool.call.completed.status`는 `success`로 고정됩니다.
- `artifact.ready`는 `final_output.solution.pdf_path`가 관측될 때 emit합니다.
- `credit.updated`는 `credit_state(balance, total_cost, difficulty)` 변경 시 emit합니다.
- `heartbeat`는 스트림이 일정 시간(기본 15초) 동안 유휴일 때 emit합니다.

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
- `ai + node=Review` -> `review`
- `ai + node=Suggestion` -> `suggestion`
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

LLM 스트림 관련 주의:
- `llm.message.completed`는 `finish_reason`으로 `stop`, `switch`, `error`를 전달할 수 있습니다.
- `llm.message.completed`는 `stream_tokens=true` 조건에서만 emit됩니다.

## message_id 규칙
- `m_llm_<n>`: LLM 토큰 스트림 단위 식별자
- `m_chat_<n>`: 완결 `chat.message` 식별자
- `m_interrupt_<n>`, `m_error_<n>`: 인터럽트/에러 메시지 식별자
- `tool_call_<n>`: tool 이벤트 fallback id

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
