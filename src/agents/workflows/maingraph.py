"""Main LangGraph definition for the Proovy agent.

상위 레벨에서 Preprocessing / Router / RAG / Features / Review
같은 큰 노드들(서브그래프) 사이의 흐름을 제어하는 메인 그래프입니다.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import AgentState
from core.llm import get_model
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
    "solve": solve_graph,
    "explain": explain_graph,
    "CreateGraph": create_graph_graph,
    "variant": variant_graph,
    "solution": solution_graph,
    "check": check_graph,
}

# --- Main Graph Nodes (서브그래프에 없는 노드들) ---


def review(state: AgentState) -> AgentState:
    # Flowchart: "Reviewer"
    """실행 결과를 검토하고 다음 단계를 결정합니다 (Pass/Fail)."""
    print("---MAIN: REVIEWING---")
    review_state = state.get("review_state")
    should_retry = False
    if review_state is not None:
        if isinstance(review_state, dict):
            passed = review_state.get("passed", True)
        else:
            passed = getattr(review_state, "passed", True)
    should_retry = not passed
    state["prev_action"] = "Review"
    state["next_action"] = "RetryCounter" if should_retry else "Suggestion"
    return state


def route_after_review(state: AgentState) -> str:
    return state.get("next_action", "Suggestion")


def suggestion(state: AgentState) -> AgentState:
    # Flowchart: "다음 학습 제안"
    """다음 학습을 제안합니다."""
    print("---MAIN: SUGGESTING NEXT STEP---")
    # TODO: 다음 학습 제안 로직 구현
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
    # TODO: 폴백 응답 로직 구현
    return state


# --- Graph Builder ---
builder = StateGraph(AgentState)

# --- 노드 등록 ---
# 1. 서브그래프 노드
builder.add_node("Preprocessing", preprocessing_graph)
builder.add_node("Router", router_graph)
builder.add_node("RAG", rag_graph)
# 2. Feature 서브그래프 노드들
for name, graph_obj in FEATURE_MAP.items():
    # name.capitalize()는 'CreateGraph' -> 'Creategraph'가 되므로, CamelCase는 그대로 써야 함
    builder.add_node(name, graph_obj)
# 3. Main 그래프 자체 노드
builder.add_node("Review", review)
builder.add_node("Suggestion", suggestion)
builder.add_node("Fallback", fallback)
builder.add_node("Simple_response", simple_response)


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
        name.capitalize(),
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
    },
)

# 8. 최종 응답 및 종료
builder.add_edge("Suggestion", END)
builder.add_edge("Fallback", END)
builder.add_edge("Simple_response", END)


# "agent": "src.agents.workflows.maingraph:graph"
graph = builder.compile()
