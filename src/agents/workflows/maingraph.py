"""Main LangGraph definition for the Proovy agent.

상위 레벨에서 Preprocessing / Router / RAG / Features / Review
같은 큰 노드들(서브그래프) 사이의 흐름을 제어하는 메인 그래프입니다.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import AgentState
from core.llm import get_model
from agents.workflows.review_logic import run_review, run_suggestion
from agents.workflows.final_response import final_response
from schema.models import OpenRouterModelName

# 각 서브그래프들을 import 합니다.
from agents.workflows.subgraphs.step1_preprocessing.graph import (
    graph as preprocessing_graph,
)
from agents.workflows.subgraphs.step2_router.graph import graph as router_graph
from agents.workflows.subgraphs.step3_rag.graph import graph as rag_graph

# step4_features에서 각 기능별 서브그래프를 모두 import 합니다.
from agents.workflows.subgraphs.step4_features import (
    check_graph,
    create_graph_graph,
    explain_graph,
    solution_graph,
    solve_graph,
    variant_graph,
)

# --- Feature 서브그래프 매핑 ---
# StepRouter의 결과('solve', 'explain' 등)와 실제 그래프 객체를 연결합니다.
FEATURE_MAP = {
    "Solve": solve_graph,
    "Explain": explain_graph,
    "CreateGraph": create_graph_graph,
    "Variant": variant_graph,
    "Solution": solution_graph,
    "Check": check_graph,
}

# Review 재시도 상한은 Router의 RetryCounter에서 관리한다.

# --- Main Graph Nodes (서브그래프에 없는 노드들) ---


def review(state: AgentState) -> AgentState:
    print("---MAIN: REVIEWING---")
    try:
        patch = run_review(state) or {}
    except Exception as exc:
        print("run_review error:", exc)
        current_retry = state.get("retry_count", 0)
        patch = {
            "review_state": {
                "passed": False,
                "feedback": "리뷰 내부 오류(자동 재시도 예정)",
                "suggestions": [],
                "retry_count": current_retry,
            }
        }

    review_state = patch.get("review_state") or state.get("review_state")
    passed = True
    retry_count = state.get("retry_count", 0) or 0
    if review_state is not None:
        if isinstance(review_state, dict):
            passed = review_state.get("passed", True)
        else:
            passed = getattr(review_state, "passed", True)
    if isinstance(review_state, dict):
        review_state["retry_count"] = retry_count

    # RetryCounter에서 state 업데이트가 누락되므로, 여기서 카운트를 관리한다.
    retry_count = state.get("retry_count", 0) or 0
    if not passed:
        retry_count += 1
    patch["retry_count"] = retry_count

    # RetryCounter가 업데이트하지 못하는 retry_limit_exceeded도 여기서 관리
    if not passed and retry_count > 2:
        patch["retry_limit_exceeded"] = True
    else:
        # 기존 값이 남지 않도록 명시적으로 false 처리
        patch["retry_limit_exceeded"] = False

    if isinstance(review_state, dict):
        review_state["retry_count"] = retry_count

    should_retry = not passed
    if patch.get("retry_limit_exceeded"):
        should_retry = False

    # meta
    patch["prev_action"] = "Review"
    if patch.get("retry_limit_exceeded"):
        patch["next_action"] = "Fallback"
    else:
        patch["next_action"] = "RetryCounter" if should_retry else "Suggestion"

    # Merge patch into state, then return full state so Studio shows updated state
    state.update(patch)
    return state


def route_after_review(state: AgentState) -> str:
    return state.get("next_action", "Suggestion")


def suggestion(state: AgentState) -> AgentState:
    print("---MAIN: SUGGESTING NEXT STEP---")
    try:
        patch = run_suggestion(state) or {}
    except Exception as exc:
        print("run_suggestion error:", exc)
        patch = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "제안 생성 중 오류가 발생했습니다. 나중에 다시 시도해주세요.",
                }
            ],
            "final_output": {"suggestion_summary": "제안 생성 실패"},
        }

    patch["prev_action"] = "Suggestion"

    # Merge messages (append) and final_output safely
    existing_messages = state.get("messages") or []
    new_messages = patch.pop("messages", [])
    state["messages"] = existing_messages + new_messages

    # Merge final_output dict
    if "final_output" in patch:
        cur_final = state.get("final_output") or {}
        updated_final = (
            dict(cur_final) if isinstance(cur_final, dict) else {"text": str(cur_final)}
        )
        pf = patch.pop("final_output")
        if isinstance(pf, dict):
            updated_final.update(pf)
        else:
            updated_final["suggestion_summary"] = str(pf)
        state["final_output"] = updated_final

    # Merge any other keys from patch
    state.update(patch)
    state["prev_action"] = "Suggestion"
    return state


def simple_response(state: AgentState) -> AgentState:
    # Flowchart: "단순 응답"
    """Router 단계에서 단순 응답으로 판별된 경우 최종 답변을 구성합니다."""
    print("---MAIN: SIMPLE RESPONSE---")
    messages = state.get("messages") or []
    user_text = ""
    if messages:
        last_message: BaseMessage = messages[-1]
        # HumanMessage 또는 user 타입인 경우에만 텍스트를 사용
        if getattr(last_message, "type", None) in {"human", "user"} and isinstance(
            getattr(last_message, "content", ""), str
        ):
            user_text = last_message.content.strip()

    if not user_text:
        # 사용자 질문이 없으면 별도 응답을 만들지 않고 그대로 반환
        return state

    system_prompt = (
        "You are a friendly Korean tutor chatbot. "
        "The user asked a non-STEM question. "
        "Answer briefly and conversationally in natural Korean, "
        "without complex math or formulas."
    )

    model = get_model(OpenRouterModelName.GPT_5_MINI)
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]

    ai_message = model.invoke(prompt_messages)

    # LangGraph state reducer(add_messages)를 이용해 새 AI 메시지를 추가
    state["messages"] = (messages or []) + [ai_message]
    state["prev_action"] = "SimpleResponse"
    return state


def fallback(state: AgentState) -> AgentState:
    # Flowchart: "폴백 응답"
    """재시도 횟수 초과 시 폴백 응답을 처리합니다."""
    print("---MAIN: FALLBACK RESPONSE---")
    retry_count = state.get("retry_count", 0)
    message = AIMessage(
        content="재시도 한도를 초과하여 폴백 응답으로 전환합니다. 질문을 조금 더 구체적으로 적어주세요."
    )
    state["messages"] = (state.get("messages") or []) + [message]
    final_output = state.get("final_output") or {}
    updated_final = (
        dict(final_output)
        if isinstance(final_output, dict)
        else {"text": str(final_output)}
    )
    updated_final["fallback"] = {
        "reason": "retry_limit_exceeded",
        "retry_count": retry_count,
    }
    state["final_output"] = updated_final
    state["prev_action"] = "Fallback"
    return state


# --- Graph Builder ---
builder = StateGraph(AgentState)

# --- 노드 등록 ---
# 1. 서브그래프 노드 (내부 LLM 호출 시 스트림 방지)
builder.add_node("Preprocessing", preprocessing_graph, tags=["nostream"])
builder.add_node("Router", router_graph, tags=["nostream"])
builder.add_node("RAG", rag_graph, tags=["nostream"])
# 2. Feature 서브그래프 노드들 (내부 LLM 호출 시 스트림 방지)
for name, graph_obj in FEATURE_MAP.items():
    builder.add_node(name, graph_obj, tags=["nostream"])
# 3. Main 그래프 자체 노드
builder.add_node("Review", review, tags=["nostream"])
builder.add_node("Suggestion", suggestion, tags=["nostream"])
builder.add_node("Fallback", fallback)
builder.add_node("Simple_response", simple_response, tags=["nostream"])
# FinalResponse 노드는 LangGraph가 내부 LLM 호출을 감지하여
# 자동으로 토큰을 스트리밍하도록 nostream 태그를 붙이지 않습니다.
builder.add_node("FinalResponse", final_response)


# --- 엣지 연결 ---

# 1. 시작점
builder.set_entry_point("Preprocessing")

# 2. Preprocessing -> Router
# Preprocessing 서브그래프는 어떤 경우든 종료 후, maingraph로 돌아옵니다.
builder.add_edge("Preprocessing", "Router")

# 3. RAG -> Router (IntentRoute)
# RAG 실행 후, Router 서브그래프의 'IntentRoute' 노드부터 다시 시작합니다.
# `configurable`을 사용하여 동적으로 시작점을 지정하는 로직이 필요하지만,
# 여기서는 개념적으로 가장 가까운 Router 노드로 다시 연결합니다.
# 실제 호출 시에는 `invoke(..., config={"start_at": "IntentRoute"})`가 사용됩니다.
builder.add_edge("RAG", "Router")


# 5. Router(StepRouter) -> Features
# Router가 'StepRouter' 노드 실행 후 종료되면, state의 'current_step'에 따라
# 해당하는 Feature 노드로 분기합니다.
def route_to_feature(state: AgentState) -> str:
    # Flowchart: "StepRouter"
    if state.get("simple_response"):
        return "Simple_response"
    if state.get("prev_action") == "Intent":
        return "RAG"
    if state.get("retry_limit_exceeded"):
        return "Fallback"

    step = state.get("current_step", "")
    if step in builder.nodes:
        return step
    # 플랜의 마지막 단계였거나, 스텝이 없는 경우
    return "Review"


builder.add_conditional_edges(
    "Router",
    route_to_feature,
    {
        "Solve": "Solve",
        "Explain": "Explain",
        "CreateGraph": "CreateGraph",
        "Variant": "Variant",
        "Solution": "Solution",
        "Check": "Check",
        "RAG": "RAG",
        "Review": "Review",
        "Fallback": "Fallback",
        "Simple_response": "Simple_response",
    },
)


# 6. Features -> Plan 완료 체크
def route_after_feature(state: AgentState) -> str:
    # Flowchart: "모든 단계 완료?"
    """Feature 실행 후, 플랜의 다음 단계가 있는지 확인합니다."""
    if not state.get("plan"):  # plan이 비어있으면
        return "Review"
    return "Router"  # 다음 단계를 위해 Executor를 다시 호출해야 함


# 각 Feature 노드 실행 후에는 route_after_feature 함수를 통해 분기합니다.
for name in FEATURE_MAP:
    builder.add_conditional_edges(
        name,
        route_after_feature,
        {
            "Router": "Router",
            "Review": "Review",
        },
    )


# 7. Review -> Suggestion or RetryCounter
builder.add_conditional_edges(
    "Review",
    route_after_review,
    {
        "Suggestion": "Suggestion",
        "RetryCounter": "Router",  # Router의 'RetryCounter' 노드 호출
        "Fallback": "Fallback",
    },
)

# 8. 최종 응답 및 종료
builder.add_edge("Suggestion", "FinalResponse")
builder.add_edge("Fallback", "FinalResponse")
builder.add_edge("Simple_response", "FinalResponse")
builder.add_edge("FinalResponse", END)


# "agent": "src.agents.workflows.maingraph:graph"
graph = builder.compile()
