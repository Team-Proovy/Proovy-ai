"""
FastAPI 서비스 엔트리포인트.

- lifespan 컨텍스트에서 LangGraph용 체크포인트/스토어/에이전트들을 초기화한다.
- /info, /invoke, /stream, /feedback, /history, /health 등의 HTTP 엔드포인트를 정의한다.
- LangGraph 에이전트 실행 결과를 이 서비스 전용 ChatMessage 스키마와
  SSE(text/event-stream) 형식으로 변환해 클라이언트에 반환한다.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import monotonic
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from core import settings
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from service.credit_service import get_credit_service, normalize_auth_token
from service.sse_v2 import (
    CHAT_MESSAGE_KIND_ASSISTANT_FINAL,
    SseV2Emitter,
    chat_kind_from_message,
    chat_role_from_type,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

# Reduce noisy INFO logs from the E2B client so sandbox
# HTTP requests don't look like errors in the service logs.
logging.getLogger("e2b.api").setLevel(logging.WARNING)
logging.getLogger("e2b.api.client_sync").setLevel(logging.WARNING)


PROGRESS_MESSAGES: dict[str, str] = {
    "CheckType": "첨부된 파일 유형을 분석하고 있습니다.",
    "FileConvert": "문서를 이미지로 변환하고 있습니다.",
    "VisionLLM": "이미지에서 텍스트와 수식을 추출하고 있습니다.",
    "Intent": "질문의 의도를 분석하고 있습니다.",
    "Planner": "여러 단계의 학습 계획을 세우고 있습니다.",
    "Executor": "계획된 단계를 실행할 준비를 하고 있습니다.",
    "RetryCounter": "이전 시도가 충분했는지 확인하고 있습니다.",
    "EmbeddingSearch": "관련 자료를 검색하고 있습니다.",
    "RelevanceCheck": "검색된 자료의 관련성을 평가하고 있습니다.",
    "RetrievedDocs": "검색된 자료를 컨텍스트에 주입하고 있습니다.",
    "Solve_Analysis": "문제를 분석하고 필요한 정보를 정리하고 있습니다.",
    "Solve_Strategy": "문제 풀이 전략과 코드를 생성하고 있습니다.",
    "Solve_Computation": "파이썬 코드를 실제로 실행 중입니다.",
    "Solve_Writer": "풀이 결과를 정리하여 답변을 작성하고 있습니다.",
    "Explain": "질문 내용을 쉽게 설명할 방법을 정리하고 있습니다.",
    "Explain_Writer": "개념 설명을 작성하고 있습니다.",
    "CreateGraph": "문제 상황을 그래프로 시각화할 방법을 고민하고 있습니다.",
    "Variant": "비슷한 유형의 변형 문제를 생성하고 있습니다.",
    "Solution": "풀이 과정을 정리하고 있습니다.",
    "Check": "답이 올바른지 검산하고 있습니다.",
    "Review": "전체 풀이 결과를 자동으로 리뷰하고 있습니다.",
    "Suggestion": "다음 학습 방향에 대한 제안을 준비하고 있습니다.",
}
CHAT_MESSAGE_EMITTING_NODES: dict[str, str] = {
    "FinalResponse": "assistant_final",
    "Simple_response": "assistant_partial",
    "CreditInsufficient": "system_notice",
    "Fallback": "system_notice",
}
VISIBLE_NODES: set[str] = set(PROGRESS_MESSAGES.keys()) | {
    "FinalResponse",
    "Simple_response",
    "Fallback",
    "CreditInsufficient",
    "CreditCheck",
    "Router",
    "Preprocessing",
    "RAG",
    "Solve",
    "Explain",
    "CreateGraph",
    "Variant",
    "Solution",
    "Check",
}
STREAM_V2_HEARTBEAT_INTERVAL_SEC = 15.0


def _normalize_node_name(node: Any) -> str:
    return str(node).split("/")[-1]


def _normalize_node_path(node_path: Any | None, node: Any) -> str:
    if node_path is None:
        return str(node)
    if isinstance(node_path, (tuple, list)):
        return "/".join(str(part) for part in node_path)
    return str(node_path)


def _coalesce_messages(new_messages: list[Any]) -> list[Any]:
    processed_messages: list[Any] = []
    current_message: dict[str, Any] = {}
    for message in new_messages:
        if isinstance(message, tuple):
            key, value = message
            current_message[key] = value
            continue
        if current_message:
            processed_messages.append(_create_ai_message(current_message))
            current_message = {}
        processed_messages.append(message)
    if current_message:
        processed_messages.append(_create_ai_message(current_message))
    return processed_messages


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(
            HTTPBearer(
                description="Please provide AUTH_SECRET api key.", auto_error=False
            )
        ),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and agents with async loading - for example for starting up MCP clients.
    """
    try:
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # store 쪽에만 setup 이 필요한 경우 호출 (없으면 무시)
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            # Checkpointer를 agents에 주입 (멀티턴 대화 지원)
            # saver가 None인 경우에도 set_checkpointer를 호출하여 agents가 checkpointer 없이 동작하도록 함
            from agents.agents import set_checkpointer

            set_checkpointer(saver)
            if saver is not None:
                logger.info("Checkpointer injected into agents")
            else:
                logger.warning(
                    "Running without checkpointer - conversation history will not be persisted"
                )

            # Configure agents with both memory components and async loading
            get_all_agent_info()

            # load_agents 필요하면 하기

            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(
    lifespan=lifespan,
    generate_unique_id_function=custom_generate_unique_id,
    root_path="/ai",
)
router = APIRouter(dependencies=[Depends(verify_bearer)])


# [중요도: 7/10] 서비스 메타데이터 조회 - 클라이언트 초기화 시 필수, 가용 에이전트/모델 목록 제공
@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def _handle_input(
    user_input: UserInput, agent: AgentGraph
) -> tuple[dict[str, Any], UUID, str]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation, run_id, and thread_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()
        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # Check for reserved keys (including 'model' even if not in configurable)
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    input: Command | dict[str, Any]
    input = {
        "messages": [HumanMessage(content=user_input.message)],
        # 이전 실행에서 체크포인터가 복원한 stale 값을 매 실행마다 초기화
        "final_output": {},
        "partial_responses": [],
    }

    if user_input.files_url:
        files = list(user_input.files_url)
        primary = files[0]
        logger.info(f"_handle_input: files_url={files}, primary={primary}")
        tool_outputs: dict[str, Any] = {
            "input_path": primary,
            "ocr_provider": {
                "name": "gemini",
                "model": "google/gemini-2.5-flash",
            },
        }
        input["tool_outputs"] = tool_outputs
        input["input_files"] = files

    if user_input.chosen_features:
        input["chosen_features"] = list(user_input.chosen_features)

    # 크레딧 잔액 조회 및 초기 상태 설정
    auth_token = normalize_auth_token(getattr(user_input, "auth_token", None))

    try:
        credit_service = get_credit_service()
        balance = await credit_service.get_balance(user_id, token=auth_token)
        input["credit_state"] = {
            "balance": balance.total_available,
            "total_cost": 0,
            "cost_per_node": {},
            "difficulty": "easy",
            "insufficient": False,
            "stopped_at_feature": None,
        }
        logger.info(f"_handle_input: credit_balance={balance.total_available}")
    except Exception as e:
        logger.warning(f"Failed to get credit balance: {e}")
        input["credit_state"] = {
            "balance": 0,
            "total_cost": 0,
            "cost_per_node": {},
            "difficulty": "easy",
            "insufficient": True,
            "stopped_at_feature": None,
        }

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id, thread_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, _thread_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(
            **kwargs, stream_mode=["updates", "values"]
        )
        response_type, response = response_events[-1]
        if response_type == "values":
            output = langchain_to_chat_message(response["messages"][-1])
            final_output = response.get("final_output")
            if final_output is not None:
                output.custom_data["final_output"] = final_output
        elif response_type == "updates" and "__interrupt__" in response:
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, thread_id = await _handle_input(user_input, agent)

    thread_id_event = {
        "type": "thread_id",
        "thread_id": thread_id,
        "run_id": str(run_id),
    }
    yield f"data: {json.dumps(thread_id_event, ensure_ascii=False)}\n\n"

    credit_usage: dict[str, Any] = {
        "used_features": [],
        "total_cost": 0,
        "difficulty": "easy",
    }

    progress_messages = PROGRESS_MESSAGES
    emitted_progress_nodes: set[str] = set()

    def emit_progress(node_name: str) -> str | None:
        if node_name in emitted_progress_nodes:
            return None
        message = progress_messages.get(node_name)
        if message is None:
            return None
        emitted_progress_nodes.add(node_name)

        # ChatMessage 형태로 생성하여 message 타입으로 전송
        progress = ChatMessage(
            type="custom",
            content="",
            custom_data={"node": node_name, "status": message},
        )
        progress.run_id = str(run_id)

        return (
            "data: "
            + json.dumps(
                {"type": "message", "content": progress.model_dump()},
                ensure_ascii=False,
            )
            + "\n\n"
        )

    try:
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue

            node_path: Any | None = None
            if len(stream_event) == 3:
                node_path, stream_mode, event = stream_event
            else:
                stream_mode, event = stream_event

            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue

                    node_name = str(node).split("/")[-1]
                    progress_line = emit_progress(node_name)
                    if progress_line is not None:
                        yield progress_line  # type: ignore[misc]

                    feature_nodes = {
                        "Solve",
                        "Explain",
                        "CreateGraph",
                        "Variant",
                        "Solution",
                        "Check",
                    }
                    feature_name = node_name
                    if node_path:
                        try:
                            path_str = str(node_path)
                            if isinstance(node_path, (tuple, list)) and node_path:
                                feature_name = str(node_path[0])
                            elif "/" in path_str:
                                feature_name = path_str.split("/")[0]
                        except Exception:
                            pass
                    if (
                        feature_name in feature_nodes
                        and feature_name not in credit_usage["used_features"]
                    ):
                        credit_usage["used_features"].append(feature_name)
                        logger.info(f"Feature executed: {feature_name}")

                    updates = updates or {}

                    if "credit_state" in updates:
                        cs = updates["credit_state"]
                        if isinstance(cs, dict):
                            credit_usage["difficulty"] = cs.get("difficulty", "easy")
                            credit_usage["total_cost"] = cs.get("total_cost", 0)

                    update_messages = updates.get("messages", [])
                    node_text = str(node)
                    if "supervisor" in node_text or "sub-agent" in node_text:
                        if not update_messages:
                            update_messages = []
                        elif isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node_text and len(update_messages) > 1:
                                update_messages = update_messages[-2:]
                            else:
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []

                    # FinalResponse 노드의 메시지 전송:
                    # - stream_tokens=True일 때: 토큰 스트리밍(messages)으로 이미 전송되므로 건너뜀
                    # - stream_tokens=False일 때: updates에서 최종 응답을 전송해야 함
                    if node_name == "FinalResponse" and update_messages:
                        if not user_input.stream_tokens:
                            new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    current_message[key] = value
                else:
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'}, ensure_ascii=False)}\n\n"
                    continue
                if (
                    chat_message.type == "human"
                    and chat_message.content == user_input.message
                ):
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()}, ensure_ascii=False)}\n\n"

            if stream_mode == "messages":
                # LangGraph가 LLM 토큰을 messages 스트림으로 전달해 줄 때,
                # 여기서는 node_path에 관계없이 토큰을 그대로 클라이언트로 전달한다.
                # (중간 노드에서의 불필요한 스트리밍은 그래프 쪽의 nostream 태그로 제어함)
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)}, ensure_ascii=False)}\n\n"

    except Exception:
        logger.exception("Error in message generator")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'}, ensure_ascii=False)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def message_generator_v2(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """Generate an SSE v2 stream with event-oriented payloads."""
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id_obj, thread_id = await _handle_input(user_input, agent)
    run_id = str(run_id_obj)
    emitter = SseV2Emitter(run_id=run_id, thread_id=thread_id)
    run_started_at = monotonic()
    heartbeat_interval_sec = STREAM_V2_HEARTBEAT_INTERVAL_SEC

    emitted_progress_nodes: set[str] = set()
    node_started_at: dict[str, float] = {}
    node_completed: set[str] = set()
    emitted_artifacts: set[str] = set()
    last_credit_signature: tuple[Any, Any, Any] | None = None

    chat_message_count = 0
    llm_message_count = 0
    active_llm_message_id: str | None = None
    active_llm_node: str | None = None
    active_llm_token_index = 0
    final_message_id: str | None = None
    terminal_emitted = False
    emitted_chat_signatures: set[str] = set()
    tool_started_at: dict[str, float] = {}

    def next_message_id(prefix: str) -> str:
        nonlocal chat_message_count
        chat_message_count += 1
        return f"{prefix}_{chat_message_count}"

    def sanitize_error_message(_: Exception) -> str:
        return "Internal server error"

    def emit_chat_events(
        raw_message: Any,
        node_name: str | None,
        kind_override: str | None = None,
    ) -> list[str]:
        nonlocal final_message_id
        events: list[str] = []
        try:
            chat_message = langchain_to_chat_message(raw_message)
            chat_message.run_id = run_id
        except Exception as e:
            logger.error(f"Error parsing v2 chat message: {e}")
            events.append(
                emitter.emit(
                    "chat.message",
                    {
                        "message_id": next_message_id("m_error"),
                        "role": "system",
                        "kind": "system_notice",
                        "content": "Unexpected error",
                    },
                )
            )
            return events

        if chat_message.type == "human" and chat_message.content == user_input.message:
            return events

        message_id = next_message_id("m_chat")
        kind = kind_override or chat_kind_from_message(
            message_type=chat_message.type,
            node_name=node_name,
            custom_data=chat_message.custom_data,
        )
        content_str = convert_message_content_to_string(chat_message.content)
        content_hash = hashlib.md5(
            content_str.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:16]
        signature = f"{chat_message.type}|{kind}|{node_name or ''}|{content_hash}"
        if signature in emitted_chat_signatures:
            return events
        emitted_chat_signatures.add(signature)
        payload: dict[str, Any] = {
            "message_id": message_id,
            "role": chat_role_from_type(chat_message.type),
            "kind": kind,
            "content": chat_message.content,
        }
        if node_name:
            payload["node"] = node_name
        events.append(emitter.emit("chat.message", payload))

        if kind == CHAT_MESSAGE_KIND_ASSISTANT_FINAL:
            final_message_id = message_id
        return events

    def emit_interrupt_event(interrupt: Any, node_name: str | None) -> str | None:
        nonlocal final_message_id
        content = str(getattr(interrupt, "value", interrupt))
        content_hash = hashlib.md5(
            content.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:16]
        signature = f"interrupt|system_notice|{node_name or ''}|{content_hash}"
        if signature in emitted_chat_signatures:
            return None
        emitted_chat_signatures.add(signature)

        message_id = next_message_id("m_interrupt")
        final_message_id = message_id
        payload: dict[str, Any] = {
            "message_id": message_id,
            "role": "assistant",
            "kind": "system_notice",
            "content": content,
        }
        if node_name:
            payload["node"] = node_name
        return emitter.emit("chat.message", payload)

    def _coerce_state_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        return {}

    def _node_name_from_event(stream_event: dict[str, Any]) -> str | None:
        metadata = stream_event.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        node = metadata_dict.get("langgraph_node") or metadata_dict.get("node")
        if node:
            return _normalize_node_name(node)

        name = stream_event.get("name")
        if isinstance(name, str):
            normalized = _normalize_node_name(name)
            if (
                normalized in PROGRESS_MESSAGES
                or normalized in CHAT_MESSAGE_EMITTING_NODES
            ):
                return normalized
        return None

    def _node_path_from_event(
        stream_event: dict[str, Any], node_name: str | None
    ) -> str:
        metadata = stream_event.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        raw_path = metadata_dict.get("langgraph_path") or metadata_dict.get("node_path")
        if isinstance(raw_path, (tuple, list)):
            path = "/".join(str(item) for item in raw_path if str(item))
            if path:
                return path
        elif isinstance(raw_path, str) and raw_path.strip():
            return raw_path
        return node_name or ""

    def _extract_finish_reason(end_data: dict[str, Any]) -> str:
        output = end_data.get("output")
        response_metadata = getattr(output, "response_metadata", None)
        if isinstance(response_metadata, dict):
            finish_reason = response_metadata.get("finish_reason")
            if finish_reason is not None:
                return str(finish_reason)
        return "stop"

    def _extract_delta_from_chunk(chunk: Any) -> str:
        if isinstance(chunk, AIMessageChunk):
            content = remove_tool_calls(chunk.content)
            if not content:
                return ""
            return convert_message_content_to_string(content)

        content = getattr(chunk, "content", None)
        if isinstance(content, (str, list)):
            filtered = remove_tool_calls(content)
            if not filtered:
                return ""
            return convert_message_content_to_string(filtered)

        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            text = chunk.get("text")
            if isinstance(text, str):
                return text
        return ""

    def _events_from_state_like(
        state_like: dict[str, Any], node_name: str | None
    ) -> list[str]:
        nonlocal last_credit_signature
        lines: list[str] = []
        credit_state = state_like.get("credit_state")
        if isinstance(credit_state, dict):
            signature = (
                credit_state.get("balance"),
                credit_state.get("total_cost"),
                credit_state.get("difficulty"),
            )
            if signature != last_credit_signature:
                last_credit_signature = signature
                balance = credit_state.get("balance", 0)
                total_cost = credit_state.get("total_cost", 0)
                remaining = 0
                if isinstance(balance, (int, float)) and isinstance(
                    total_cost, (int, float)
                ):
                    remaining = balance - total_cost
                lines.append(
                    emitter.emit(
                        "credit.updated",
                        {
                            "balance": balance,
                            "total_cost": total_cost,
                            "remaining": remaining,
                        },
                    )
                )

        final_output = state_like.get("final_output")
        solution_output = (
            final_output.get("solution") if isinstance(final_output, dict) else None
        )
        if isinstance(solution_output, dict):
            artifact_path = solution_output.get("pdf_path")
            if artifact_path:
                artifact_id = str(solution_output.get("pdf_file_name") or artifact_path)
                if artifact_id not in emitted_artifacts:
                    emitted_artifacts.add(artifact_id)
                    lines.append(
                        emitter.emit(
                            "artifact.ready",
                            {
                                "artifact_id": artifact_id,
                                "name": solution_output.get("pdf_file_name")
                                or artifact_id,
                                "mime": solution_output.get(
                                    "pdf_mime_type", "application/pdf"
                                ),
                                "path": artifact_path,
                                "size": solution_output.get("pdf_file_size", 0),
                            },
                        )
                    )

        emit_kind = CHAT_MESSAGE_EMITTING_NODES.get(node_name or "")
        if emit_kind:
            messages = state_like.get("messages")
            if isinstance(messages, list) and messages:
                final_candidate = messages[-1]
                for sse_line in emit_chat_events(
                    final_candidate,
                    node_name,
                    kind_override=emit_kind,
                ):
                    lines.append(sse_line)

        return lines

    async def _pump_stream(queue: asyncio.Queue[tuple[str, Any]]) -> None:
        try:
            async for stream_event in agent.astream_events(
                kwargs["input"],
                kwargs["config"],
                version="v2",
            ):
                await queue.put(("event", stream_event))
        except Exception as exc:
            await queue.put(("error", exc))
        finally:
            await queue.put(("done", None))

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    pump_task = asyncio.create_task(_pump_stream(queue))

    try:
        yield emitter.emit(
            "session.metadata",
            {
                "agent_id": agent_id,
                "capabilities": {
                    "token_stream": True,
                    "heartbeat": True,
                    "terminal_event": True,
                },
            },
        )
        yield emitter.emit(
            "run.started", {"stream_tokens": bool(user_input.stream_tokens)}
        )

        while True:
            try:
                item_type, item = await asyncio.wait_for(
                    queue.get(), timeout=heartbeat_interval_sec
                )
            except asyncio.TimeoutError:
                yield emitter.emit("heartbeat", {"alive": True})
                continue

            if item_type == "done":
                break
            if item_type == "error":
                raise item

            stream_event = item
            if not isinstance(stream_event, dict):
                continue

            event_name = str(stream_event.get("event") or "")
            data = stream_event.get("data")
            data_dict = data if isinstance(data, dict) else {}
            metadata = stream_event.get("metadata")
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            tags = stream_event.get("tags")
            tag_list = tags if isinstance(tags, list) else []

            node_name = _node_name_from_event(stream_event)
            node_path_str = _node_path_from_event(stream_event, node_name)

            if event_name == "on_chain_start" and node_name:
                if node_name not in VISIBLE_NODES:
                    continue
                if node_name not in node_started_at:
                    node_started_at[node_name] = monotonic()
                    yield emitter.emit(
                        "node.started",
                        {"node": node_name, "node_path": node_path_str},
                    )
                    progress = PROGRESS_MESSAGES.get(node_name)
                    if progress and node_name not in emitted_progress_nodes:
                        emitted_progress_nodes.add(node_name)
                        yield emitter.emit(
                            "node.progress",
                            {"node": node_name, "message": progress},
                        )
                continue

            if event_name == "on_chain_stream":
                state_patch = _coerce_state_dict(data_dict.get("chunk"))
                if state_patch:
                    if isinstance(state_patch.get("__interrupt__"), list):
                        interrupt: Interrupt
                        for interrupt in state_patch["__interrupt__"]:
                            interrupt_event = emit_interrupt_event(interrupt, node_name)
                            if interrupt_event:
                                yield interrupt_event
                    for line in _events_from_state_like(state_patch, node_name):
                        yield line
                continue

            if event_name == "on_chain_end" and node_name:
                state_output = _coerce_state_dict(data_dict.get("output"))
                if state_output:
                    if isinstance(state_output.get("__interrupt__"), list):
                        interrupt: Interrupt
                        for interrupt in state_output["__interrupt__"]:
                            interrupt_event = emit_interrupt_event(interrupt, node_name)
                            if interrupt_event:
                                yield interrupt_event
                    for line in _events_from_state_like(state_output, node_name):
                        yield line

                if node_name not in VISIBLE_NODES:
                    continue
                if node_name not in node_completed:
                    node_completed.add(node_name)
                    started_at = node_started_at.get(node_name, monotonic())
                    duration_ms = max(0, int((monotonic() - started_at) * 1000))
                    yield emitter.emit(
                        "node.completed",
                        {
                            "node": node_name,
                            "status": "success",
                            "duration_ms": duration_ms,
                        },
                    )
                continue

            if event_name == "on_custom_event":
                custom_payload = stream_event.get("data")
                if isinstance(custom_payload, dict):
                    custom_message = ChatMessage(
                        type="custom",
                        content="",
                        custom_data=custom_payload,
                    )
                    for line in emit_chat_events(custom_message, node_name):
                        yield line
                continue

            if event_name == "on_tool_start":
                tool_call_id = str(
                    stream_event.get("run_id") or next_message_id("tool_call")
                )
                tool_started_at[tool_call_id] = monotonic()
                tool_name = str(stream_event.get("name") or "tool")
                yield emitter.emit(
                    "tool.call.started",
                    {"tool_call_id": tool_call_id, "tool_name": tool_name},
                )
                continue

            if event_name == "on_tool_end":
                tool_call_id = str(
                    stream_event.get("run_id") or next_message_id("tool_call")
                )
                started_at = tool_started_at.pop(tool_call_id, monotonic())
                duration_ms = max(0, int((monotonic() - started_at) * 1000))
                yield emitter.emit(
                    "tool.call.completed",
                    {
                        "tool_call_id": tool_call_id,
                        "status": "success",
                        "duration_ms": duration_ms,
                    },
                )
                continue

            if event_name == "on_chat_model_start":
                if not user_input.stream_tokens:
                    continue
                if "skip_stream" in tag_list:
                    continue

                token_node = str(
                    metadata_dict.get("langgraph_node")
                    or metadata_dict.get("node")
                    or node_name
                    or active_llm_node
                    or "unknown"
                )

                if token_node in VISIBLE_NODES and token_node not in node_started_at:
                    node_started_at[token_node] = monotonic()
                    yield emitter.emit(
                        "node.started",
                        {"node": token_node, "node_path": token_node},
                    )
                    progress = PROGRESS_MESSAGES.get(token_node)
                    if progress and token_node not in emitted_progress_nodes:
                        emitted_progress_nodes.add(token_node)
                        yield emitter.emit(
                            "node.progress",
                            {"node": token_node, "message": progress},
                        )

                if active_llm_message_id is not None:
                    yield emitter.emit(
                        "llm.message.completed",
                        {
                            "message_id": active_llm_message_id,
                            "finish_reason": "switch",
                        },
                    )

                llm_message_count += 1
                active_llm_message_id = f"m_llm_{llm_message_count}"
                active_llm_node = token_node
                active_llm_token_index = 0
                yield emitter.emit(
                    "llm.message.started",
                    {
                        "message_id": active_llm_message_id,
                        "node": token_node,
                        "role": "assistant",
                    },
                )
                continue

            if event_name == "on_chat_model_stream":
                if not user_input.stream_tokens:
                    continue
                if "skip_stream" in tag_list:
                    continue

                token_node = str(
                    metadata_dict.get("langgraph_node")
                    or metadata_dict.get("node")
                    or active_llm_node
                    or "unknown"
                )
                if active_llm_message_id is None:
                    llm_message_count += 1
                    active_llm_message_id = f"m_llm_{llm_message_count}"
                    active_llm_node = token_node
                    active_llm_token_index = 0
                    yield emitter.emit(
                        "llm.message.started",
                        {
                            "message_id": active_llm_message_id,
                            "node": token_node,
                            "role": "assistant",
                        },
                    )

                delta = _extract_delta_from_chunk(data_dict.get("chunk"))
                if not delta:
                    continue

                yield emitter.emit(
                    "llm.token.delta",
                    {
                        "message_id": active_llm_message_id,
                        "node": token_node,
                        "delta": delta,
                        "index": active_llm_token_index,
                    },
                )
                active_llm_token_index += 1
                continue

            if event_name == "on_chat_model_end":
                if not user_input.stream_tokens:
                    continue
                if "skip_stream" in tag_list:
                    continue
                if active_llm_message_id is not None:
                    yield emitter.emit(
                        "llm.message.completed",
                        {
                            "message_id": active_llm_message_id,
                            "finish_reason": _extract_finish_reason(data_dict),
                        },
                    )
                    active_llm_message_id = None
                    active_llm_node = None
                    active_llm_token_index = 0
                continue

    except Exception as e:
        logger.exception("Error in message generator v2")
        if active_llm_message_id is not None:
            yield emitter.emit(
                "llm.message.completed",
                {
                    "message_id": active_llm_message_id,
                    "finish_reason": "error",
                },
            )
            active_llm_message_id = None
        if not terminal_emitted:
            terminal_emitted = True
            yield emitter.emit(
                "run.failed",
                {
                    "code": "internal_error",
                    "message": sanitize_error_message(e),
                    "retryable": False,
                },
            )
    finally:
        if not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass

        if not terminal_emitted:
            if active_llm_message_id is not None:
                yield emitter.emit(
                    "llm.message.completed",
                    {
                        "message_id": active_llm_message_id,
                        "finish_reason": "stop",
                    },
                )
            terminal_emitted = True
            duration_ms = int((monotonic() - run_started_at) * 1000)
            payload: dict[str, Any] = {"duration_ms": duration_ms}
            if final_message_id:
                payload["final_message_id"] = final_message_id
            yield emitter.emit("run.completed", payload)


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


# [중요도: 10/10] 실시간 스트리밍 - 최고 우선순위, 중간 과정/토큰 단위 응답으로 UX 극대화def _sse_response_example() -> dict[int | str, Any]:
def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post(
    "/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
async def stream(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


def _sse_v2_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response (Protocol v2)",
            "content": {
                "text/event-stream": {
                    "example": (
                        "id: 8d1f:1\n"
                        "event: session.metadata\n"
                        'data: {"v":"2.0","seq":1,"run_id":"8d1f","thread_id":"a21c"}\n\n'
                        "id: 8d1f:2\n"
                        "event: run.completed\n"
                        'data: {"v":"2.0","seq":2,"run_id":"8d1f","thread_id":"a21c","duration_ms":1200}\n\n'
                    ),
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream/v2",
    response_class=StreamingResponse,
    responses=_sse_v2_response_example(),
    operation_id="stream_v2_with_agent_id",
)
@router.post(
    "/stream/v2",
    response_class=StreamingResponse,
    responses=_sse_v2_response_example(),
)
async def stream_v2(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """Stream SSE protocol v2 events.

    v2 keeps `/stream` untouched and introduces event-oriented semantics at
    `/stream/v2` for explicit client-side state handling.
    """
    return StreamingResponse(
        message_generator_v2(user_input, agent_id),
        media_type="text/event-stream",
    )


# [중요도: 6/10] 피드백 수집 - 선택적 기능, LangSmith 모델 개선/모니터링용 래퍼
@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


# [중요도: 7/10] 대화 이력 조회 - 디버깅/UI 히스토리 표시에 유용, 자주 사용되는 보조 기능
@router.post("/history")
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [
            langchain_to_chat_message(m) for m in messages
        ]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


# [중요도: 8/10] 헬스 체크 - 프로덕션 필수, 모니터링/로드밸런서/오케스트레이션 도구에서 사용
@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = (
                "connected" if langfuse.auth_check() else "disconnected"
            )
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)


# GET /info

# 사용 가능한 에이전트 목록, 사용 가능한 LLM 모델 목록, 기본 에이전트/모델을 반환하는 메타데이터 조회용 엔드포인트.
# POST /invoke

# 기본 에이전트(DEFAULT_AGENT)에 대해 한 번 추론을 수행하고, 최종 한 개의 ChatMessage만 JSON으로 반환.
# POST /{agent_id}/invoke

# 경로로 전달한 agent_id 에이전트를 대상으로 위와 동일하게 한 번 추론을 수행하고, 최종 ChatMessage 하나를 반환.
# POST /stream

# 기본 에이전트를 대상으로, 에이전트 실행 중 나오는 중간 메시지/토큰을 Server-Sent Events(text/event-stream) 형식으로 스트리밍.
# POST /{agent_id}/stream

# 특정 agent_id 에이전트에 대해 위와 동일하게 SSE 스트리밍을 수행.
# POST /stream/v2

# v2 프로토콜(event/id/data + envelope) 기반 SSE 스트리밍.
# POST /{agent_id}/stream/v2

# 특정 agent_id 에이전트에 대해 v2 SSE 스트리밍 수행.
# POST /feedback

# LangSmith에 피드백(run_id, key, score, 추가 kwargs)을 기록하는 래퍼 엔드포인트.
# 클라이언트가 LangSmith 자격증명을 직접 가지지 않아도 서버를 통해 피드백 전송.
# POST /history

# 주어진 thread_id에 대한 대화 히스토리를 LangGraph 상태에서 읽어와 ChatHistory(메시지 배열)로 반환.
# 현재는 DEFAULT_AGENT의 상태만 조회하도록 구현.
# GET /health

# 서비스의 헬스 체크용 엔드포인트.
# 기본 "status": "ok"를 반환하고, Langfuse tracing이 활성화된 경우 Langfuse 연결 상태("connected"/"disconnected")도 함께 반환.

# The LangGraph Studio is available at /studio
# This is a placeholder for any future documentation or information about the studio.
