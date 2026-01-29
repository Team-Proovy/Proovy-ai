"""
FastAPI 서비스 엔트리포인트.

- lifespan 컨텍스트에서 LangGraph용 체크포인트/스토어/에이전트들을 초기화한다.
- /info, /invoke, /stream, /feedback, /history, /health 등의 HTTP 엔드포인트를 정의한다.
- LangGraph 에이전트 실행 결과를 이 서비스 전용 ChatMessage 스키마와
  SSE(text/event-stream) 형식으로 변환해 클라이언트에 반환한다.
"""

import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
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

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

# Reduce noisy INFO logs from the E2B client so sandbox
# HTTP requests don't look like errors in the service logs.
logging.getLogger("e2b.api").setLevel(logging.WARNING)
logging.getLogger("e2b.api.client_sync").setLevel(logging.WARNING)


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
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            # Configure agents with both memory components and async loading
            get_all_agent_info()

            # load_agents 필요하면 하기

            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
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
) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
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

    # 현재는 LangGraph checkpointer(메모리)를 붙이지 않았으므로
    # 상태 조회/인터럽트 재개는 사용하지 않고, 항상 새 메시지로만 호출한다.
    input: Command | dict[str, Any]
    input = {"messages": [HumanMessage(content=user_input.message)]}

    # files_url 이 설정된 경우, Preprocessing 이 바로 사용할 수 있도록
    # tool_outputs.input_path / input_files 를 초기 state 에 심어준다.
    if user_input.files_url:
        files = list(user_input.files_url)
        primary = files[0]
        # 디버그용 로그: files_url 이 실제로 들어오는지 확인
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

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


# [중요도: 9/10] 단일 추론 실행 - 핵심 기능, 최종 응답만 반환하는 심플한 요청-응답 패턴
@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
            # 최종 상태에 저장된 구조화 결과(final_output)를 함께 포함
            final_output = response.get("final_output")
            if final_output is not None:
                output.custom_data["final_output"] = final_output
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
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

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    # 노드별 진행 상태 문구 매핑 (최종 응답 전까지 "~하고 있습니다" 형태로 전달)
    progress_messages: dict[str, str] = {
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
        "Solve_Computation": "문제에 대한 코드를 실행 중입니다.",
        "Explain": "질문 내용을 쉽게 설명할 방법을 정리하고 있습니다.",
        "CreateGraph": "문제 상황을 그래프로 시각화할 방법을 고민하고 있습니다.",
        "Variant": "비슷한 유형의 변형 문제를 생성하고 있습니다.",
        "Solution": "풀이 과정을 정리하고 있습니다.",
        "Check": "답이 올바른지 검산하고 있습니다.",
        "Review": "전체 풀이 결과를 자동으로 리뷰하고 있습니다.",
        "Suggestion": "다음 학습 방향에 대한 제안을 준비하고 있습니다.",
    }

    def is_final_node(path: Any) -> bool:
        """스트리밍 이벤트의 node_path가 FinalResponse 노드를 가리키는지 확인.

        node_path의 실제 타입(tuple, list, NodePath, str 등)에 상관없이
        문자열 표현 안에 "FinalResponse"가 포함되어 있으면 최종 응답 노드로 간주한다.
        """
        if path is None:
            return False
        try:
            return "FinalResponse" in str(path)
        except Exception:
            return False

    def emit_progress(node_name: str | None) -> None:
        """비 최종 노드용 진행 상황 custom 메시지를 SSE로 전송."""
        if not node_name:
            return
        text = progress_messages.get(node_name)
        if not text:
            return
        progress = ChatMessage(
            type="custom", content="", custom_data={"node": node_name, "status": text}
        )  # type: ignore[call-arg]
        progress.run_id = str(run_id)
        yield_line = f"data: {json.dumps({'type': 'message', 'content': progress.model_dump()}, ensure_ascii=False)}\n\n"
        # message_generator는 async generator이므로, 내부 헬퍼에서 직접 yield 할 수 없어
        # 호출 측에서 이 문자열을 다시 yield 하도록 반환 값 대신 클로저 형태로 사용한다.
        return yield_line  # type: ignore[return-value]

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            node_path: Any | None = None
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                node_path, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue

                    # 비 최종 노드에 대해서는 진행 상황 custom 메시지만 보내고,
                    # 중간 AI/Human 메시지는 클라이언트에 노출하지 않는다.
                    node_name = str(node).split("/")[-1]
                    progress_line = emit_progress(node_name)
                    if progress_line is not None:
                        yield progress_line  # type: ignore[misc]

                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []

                    # 중간 노드의 update_messages는 클라이언트로 전달하지 않기 위해 버린다.
                    # 최종 응답 노드의 update_messages도 토큰 스트림으로 충분하므로 별도 전송하지 않는다.

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            # updates/custom에서 생성된 processed_messages만 일반 message로 전송
            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'}, ensure_ascii=False)}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
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
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'}, ensure_ascii=False)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


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
