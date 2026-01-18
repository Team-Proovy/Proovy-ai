"""Router subgraph.

사용자의 의도를 파악하고, RAG를 거쳐 실행 계획을 수립한 뒤,
각 단계를 실행할 Feature로 라우팅하고, 재시도 로직을 관리합니다.

[Flow]
1. Intent → (maingraph: RAG or END)
2. (maingraph) → IntentRoute → Planner? → Executor
3. Executor → StepRouter → (maingraph: Features)
4. (maingraph) → RetryCounter → Executor or END
"""

from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import AgentState
from core.llm import get_model
from schema.models import OpenRouterModelName

MAX_RETRIES = 2


FEATURE_ACTIONS = {
    "Solve",
    "Explain",
    "Create_graph",
    "Variant",
    "Solution",
    "Check",
}


def intent(state: AgentState) -> AgentState:
    """사용자 질문의 의도를 파악하고, RAG 호출 필요 여부 등을 결정합니다.
    어떤 경우든 router 그래프는 여기서 종료되고, maingraph가 다음을 결정합니다.
    """
    print("---ROUTER: INTENT DETECTION---")
    latest_question = ""
    messages = state.get("messages") or []
    if messages:
        last_message: BaseMessage = messages[-1]
        content = getattr(last_message, "content", "")
        latest_question = content.strip() if isinstance(content, str) else ""

    if latest_question:
        classifier = get_model(OpenRouterModelName.GPT_5_MINI)
        system_prompt = (
            "You are a strict classifier. "
            "Return only 'STEM' if the user's question is about math, physics, "
            "chemistry, biology, engineering, computer science, or similar STEM subjects. "
            "Otherwise return 'NON_STEM'."
        )
        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Question:\n"
                    f"{latest_question}\n\n"
                    "Answer with either STEM or NON_STEM."
                )
            ),
        ]
        try:
            ai_message = classifier.invoke(prompt_messages)
            verdict = getattr(ai_message, "content", "")
            normalized = (verdict or "").strip().upper()
            is_stem = normalized.startswith("STEM")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"---ROUTER: STEM CLASSIFIER ERROR {exc!r}---")
            is_stem = True

        state["simple_response"] = not is_stem
        label = "STEM" if is_stem else "NON_STEM"
        print(f"---ROUTER: STEM CLASSIFIER RESULT {label}---")
    else:
        state["simple_response"] = None
    state["prev_action"] = "Intent"
    return state


def intent_route(state: AgentState) -> Literal["Planner", "Executor"]:
    """RAG 검색 결과와 사용자 의도를 종합하여 단일/복합 의도를 결정합니다.
    - 복합 의도: 실행 계획 수립을 위해 Planner로 이동
    - 단일 의도: 바로 실행을 위해 Executor로 이동
    """
    print("---ROUTER: INTENT ROUTING---")
    # TODO: 실제 단일/복합 의도 분기 로직
    # if state.get("intent_result") == "complex":
    #     return "Planner"
    # return "Executor"
    return "Planner"  # 현재는 Planner로 고정


def planner(state: AgentState) -> AgentState:
    """복합 의도에 대한 실행 계획을 수립합니다."""
    print("---ROUTER: PLANNING---")
    # TODO: 실행 계획 수립 로직
    # state["plan"] = ["step1: solve", "step2: explain"]
    # state["retry_count"] = 0
    return state


def executor(state: AgentState) -> AgentState:
    """수립된 계획의 다음 단계를 실행 준비합니다."""
    print("---ROUTER: EXECUTING STEP---")
    # TODO: 계획에서 다음 단계(step)를 꺼내 state에 저장하는 로직
    # plan = state.get("plan", [])
    # state["current_step"] = plan.pop(0)
    return state


def step_router(state: AgentState) -> Literal["__end__"]:
    """실행할 단계를 Feature 서브그래프로 라우팅하기 위해 그래프를 종료합니다.
    maingraph는 state의 'current_step'을 보고 적절한 Feature 서브그래프를 호출합니다.
    """
    print("---ROUTER: ROUTING TO FEATURE---")
    state["prev_action"] = "StepRouter"
    return "__end__"


def retry_counter(state: AgentState) -> Literal["Executor", "__end__"]:
    """재시도 횟수를 확인하고, 다음 단계를 결정합니다.
    - 재시도 가능: Executor로 돌아가 다시 실행
    - 재시도 불가: Fallback 응답을 위해 그래프 종료
    """
    print("---ROUTER: RETRY COUNTER---")
    retries = state.get("retry_count", 0) + 1
    state["retry_count"] = retries
    if retries > MAX_RETRIES:
        state["retry_limit_exceeded"] = True
        print(f"---ROUTER: RETRY LIMIT EXCEEDED ({retries - 1})---")
        return "__end__"
    state.pop("retry_limit_exceeded", None)
    print(f"---ROUTER: RETRYING ({retries - 1}/{MAX_RETRIES})---")
    return "Executor"


def router_entry(
    state: AgentState,
) -> Literal["Intent", "IntentRoute", "RetryCounter", "Executor"]:
    """
    현재 state를 보고 어느 단계로 진입할지 결정하는 관문 노드
    """
    next_action = state.get("next_action")
    if next_action in {"Intent", "IntentRoute", "RetryCounter", "Executor"}:
        state.pop("next_action", None)
        return next_action  # 명시 지정 우선

    prev_action = state.get("prev_action")
    if prev_action in FEATURE_ACTIONS:
        return "Executor"

    if prev_action == "Preprocessing":
        return "Intent"
    if prev_action == "RAG":
        return "IntentRoute"
    if prev_action == "Review":
        return "RetryCounter"

    return "Intent"


# 그래프 구성
builder = StateGraph(AgentState)

# 노드 등록
builder.add_node("RouterEntry", lambda state: state)
builder.add_node("Intent", intent)
builder.add_node(
    "IntentRoute", lambda state: state
)  # 분기 시작점 역할만 하는 더미 노드
builder.add_node("Planner", planner)
builder.add_node("Executor", executor)
builder.add_node("StepRouter", step_router)
builder.add_node("RetryCounter", lambda state: state)  # 재시도 분기 시작점

# 1. RouterEntry: maingraph가 어떤 단계로 진입할지 결정합니다.
# state["next_action"] 값에 따라 Intent/IntentRoute/RetryCounter 중 하나로 이동합니다.
builder.set_entry_point("RouterEntry")
builder.add_conditional_edges(
    "RouterEntry",
    router_entry,
    {
        "Intent": "Intent",
        "IntentRoute": "IntentRoute",
        "RetryCounter": "RetryCounter",
        "Executor": "Executor",
    },
)

# RouterEntry가 Intent로 향한 경우, Preprocessing 단계 이후 의도 분석만 수행하고 종료합니다.
builder.add_edge("Intent", END)

# 2. RAG 후 진입점: IntentRoute에서 분기
# maingraph는 RAG 호출 후, 'IntentRoute' 노드부터 이 그래프를 다시 실행합니다.
builder.add_conditional_edges(
    "IntentRoute",
    intent_route,
    {
        "Planner": "Planner",
        "Executor": "Executor",
    },
)

# 3. 계획 수립 및 실행
builder.add_edge("Planner", "Executor")
builder.add_edge("Executor", "StepRouter")

# 4. Feature 라우팅을 위한 종료
# StepRouter는 항상 END를 향하며, maingraph가 다음 Feature 서브그래프를 결정합니다.
builder.add_edge("StepRouter", END)

# 5. 재시도(Retry) 진입점
# maingraph는 Reviewer가 'Fail'을 반환하면 'RetryCounter' 노드부터 실행합니다.
builder.add_conditional_edges(
    "RetryCounter",
    retry_counter,
    {
        "Executor": "Executor",  # 재시도
        "__end__": END,  # 재시도 횟수 초과
    },
)

graph = builder.compile()
