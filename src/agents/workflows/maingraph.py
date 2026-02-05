"""Main LangGraph definition for the Proovy agent.

상위 레벨에서 Preprocessing / Router / RAG / Features / Review
같은 큰 노드들(서브그래프) 사이의 흐름을 제어하는 메인 그래프입니다.

크레딧 관리 구조:
- 시작: 클라이언트에서 credit_state.balance 전달
- 중간: 각 Feature 노드 실행 후 Conditional Edge로 잔액 체크
- 종료: 최종 total_cost를 클라이언트에 반환 → Spring API로 정산
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import AgentState, CreditState
from core.llm import get_model
from agents.workflows.review_logic import run_review, run_suggestion
from agents.workflows.final_response import final_response
from agents.workflows.utils import (
    check_credit_sufficient,
    get_difficulty_from_state,
    COST_PER_1K_TOKENS,
)
from schema.models import OpenRouterModelName


# --- 기능별 기본 비용 및 난이도 배수 ---
FEATURE_BASE_COST = {
    "Solve": 10,
    "Explain": 5,
    "CreateGraph": 5,
    "Variant": 5,
    "Solution": 20,
    "Check": 3,
}

DIFFICULTY_MULTIPLIER = {
    "easy": 1.0,
    "medium": 1.5,
    "hard": 2.0,
}

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


# --- 크레딧 관련 노드 ---

def _calculate_feature_cost(feature_name: str, difficulty: str) -> float:
    """기능과 난이도에 따른 비용을 계산합니다."""
    base_cost = FEATURE_BASE_COST.get(feature_name, 5)
    multiplier = DIFFICULTY_MULTIPLIER.get(difficulty.lower() if difficulty else "easy", 1.0)
    return round(base_cost * multiplier, 2)


def credit_check_after_feature(state: AgentState) -> AgentState:
    """
    Feature 실행 완료 후 크레딧을 차감하고, 다음 실행을 위한 잔액을 확인합니다.
    이 노드는 각 Feature 실행 후에 호출됩니다.
    """
    print("---MAIN: CREDIT CHECK AFTER FEATURE---")

    # 방금 실행된 Feature 확인 (prev_action에서)
    prev_action = state.get("prev_action", "")

    # credit_state 가져오기 또는 초기화
    credit_state = state.get("credit_state")
    if credit_state is None:
        credit_state = {"balance": 0.0, "total_cost": 0.0, "cost_per_node": {}, "difficulty": "easy", "insufficient": False}
    elif not isinstance(credit_state, dict):
        credit_state = credit_state.model_dump() if hasattr(credit_state, 'model_dump') else {}

    difficulty = get_difficulty_from_state(state)
    credit_state["difficulty"] = difficulty

    # 실행된 Feature에 대한 비용 계산 및 기록
    if prev_action in FEATURE_BASE_COST:
        cost = _calculate_feature_cost(prev_action, difficulty)
        credit_state["total_cost"] = credit_state.get("total_cost", 0) + cost

        # 노드별 비용 기록
        cost_per_node = credit_state.get("cost_per_node", {})
        cost_per_node[prev_action] = cost_per_node.get(prev_action, 0) + cost
        credit_state["cost_per_node"] = cost_per_node

        balance = credit_state.get("balance", 0)
        total_cost = credit_state["total_cost"]
        print(f"→ Credit used: {prev_action} cost={cost}, total={total_cost}/{balance}")

    # 다음 Feature 실행을 위한 잔액 확인
    remaining_plan = state.get("plan") or []
    if remaining_plan:
        next_feature = remaining_plan[0] if remaining_plan else None
        if next_feature and next_feature in FEATURE_BASE_COST:
            next_cost = _calculate_feature_cost(next_feature, difficulty)

            # 잔액 확인 (balance - total_cost)
            balance = credit_state.get("balance", 0)
            total_cost = credit_state.get("total_cost", 0)
            available = balance - total_cost

            if available < next_cost:
                print(f"→ Insufficient credit for {next_feature} (need: {next_cost}, available: {available})")
                credit_state["insufficient"] = True
                credit_state["stopped_at_feature"] = next_feature

    state["credit_state"] = credit_state
    return state


def credit_insufficient(state: AgentState) -> AgentState:
    """크레딧 부족으로 실행을 중단합니다."""
    print("---MAIN: CREDIT INSUFFICIENT---")

    credit_state = state.get("credit_state") or {}
    if not isinstance(credit_state, dict):
        credit_state = credit_state.model_dump() if hasattr(credit_state, 'model_dump') else {}

    stopped_feature = credit_state.get("stopped_at_feature", "다음 기능")
    total_cost = credit_state.get("total_cost", 0)
    cost_per_node = credit_state.get("cost_per_node", {})
    balance = credit_state.get("balance", 0)

    # 사용 내역 문자열 생성
    usage_summary = ", ".join([f"{k}: {v}" for k, v in cost_per_node.items()]) if cost_per_node else "없음"

    message = AIMessage(
        content=(
            f"크레딧이 부족하여 '{stopped_feature}' 기능을 실행할 수 없습니다.\n\n"
            f"잔액: {balance}\n"
            f"지금까지 사용한 크레딧: {total_cost}\n"
            f"실행된 기능: {usage_summary}\n\n"
            "크레딧을 충전하시거나, 더 간단한 기능을 선택해 주세요."
        )
    )
    state["messages"] = (state.get("messages") or []) + [message]

    # final_output 업데이트
    final_output = state.get("final_output") or {}
    updated_final = (
        dict(final_output)
        if isinstance(final_output, dict)
        else {"text": str(final_output)}
    )
    updated_final["credit_insufficient"] = {
        "stopped_at": stopped_feature,
        "balance": balance,
        "total_cost": total_cost,
        "cost_per_node": cost_per_node,
    }
    state["final_output"] = updated_final
    state["prev_action"] = "CreditInsufficient"

    return state


# --- Graph Builder ---
builder = StateGraph(AgentState)

# --- 노드 등록 ---
# 1. 서브그래프 노드 (내부 LLM 호출 시 스트림 방지)
builder.add_node("Preprocessing", preprocessing_graph, tags=["nostream"])
builder.add_node("Router", router_graph, tags=["nostream"])
builder.add_node("RAG", rag_graph, tags=["nostream"])
# 2. Feature 서브그래프 노드들
# 내부 Writer 노드가 스트리밍할 수 있도록 nostream 태그를 붙이지 않습니다.
# 대신 각 서브그래프 내부에서 노드별로 skip_stream 태그를 설정합니다.
for name, graph_obj in FEATURE_MAP.items():
    builder.add_node(name, graph_obj)
# 3. Main 그래프 자체 노드
builder.add_node("Review", review, tags=["nostream"])
builder.add_node("Suggestion", suggestion, tags=["nostream"])
builder.add_node("Fallback", fallback)
builder.add_node("Simple_response", simple_response)
# 4. 크레딧 관련 노드
builder.add_node("CreditCheck", credit_check_after_feature, tags=["nostream"])
builder.add_node("CreditInsufficient", credit_insufficient)
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


# 6. Features -> CreditCheck -> Plan 완료 체크
# 각 Feature 노드 실행 후에는 CreditCheck 노드를 거칩니다.
for name in FEATURE_MAP:
    builder.add_edge(name, "CreditCheck")


def route_after_credit_check(state: AgentState) -> str:
    """CreditCheck 후 다음 단계를 결정합니다."""
    # 크레딧 부족 확인
    credit_state = state.get("credit_state")
    if credit_state:
        insufficient = False
        if isinstance(credit_state, dict):
            insufficient = credit_state.get("insufficient", False)
        else:
            insufficient = getattr(credit_state, "insufficient", False)

        if insufficient:
            return "CreditInsufficient"

    # plan이 비어있으면 Review로
    if not state.get("plan"):
        return "Review"

    # 다음 단계를 위해 Router로
    return "Router"


builder.add_conditional_edges(
    "CreditCheck",
    route_after_credit_check,
    {
        "Router": "Router",
        "Review": "Review",
        "CreditInsufficient": "CreditInsufficient",
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
builder.add_edge("CreditInsufficient", "FinalResponse")
builder.add_edge("FinalResponse", END)


# "agent": "src.agents.workflows.maingraph:graph"
graph = builder.compile()
