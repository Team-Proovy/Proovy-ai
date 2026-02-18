"""Router subgraph.

사용자의 의도를 파악하고, RAG를 거쳐 실행 계획을 수립한 뒤,
각 단계를 실행할 Feature로 라우팅하고, 재시도 로직을 관리합니다.

[Flow]
1. Intent → (maingraph: RAG or END)
2. (maingraph) → IntentRoute → Planner? → Executor
3. Executor → StepRouter → (maingraph: Features)
4. (maingraph) → RetryCounter → Executor or END
"""

import json
from typing import Literal, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.prompts.router_prompts import (
    COMPLEXITY_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    STEM_CLASSIFIER_SYSTEM_PROMPT,
    build_complexity_user_prompt,
    build_planner_hints_section,
    build_planner_output_instruction,
    build_planner_question_section,
    build_primary_feature_system_prompt,
    build_primary_feature_user_prompt,
    build_stem_context_section,
    build_stem_user_prompt,
)
from agents.state import AgentState
from agents.workflows.problem_utils import (
    extract_problem_inventory,
    input_files_fingerprint,
)
from agents.workflows.retry_trace import env_truthy, retry_trace_log
from agents.workflows.utils import extract_ocr_text
from core.llm import get_model
from schema.models import OpenRouterModelName

MAX_RETRIES = 2
FORCE_RETRY_EXECUTOR_TO_REVIEW_ENV = "FORCE_RETRY_EXECUTOR_TO_REVIEW"


FEATURE_ACTIONS = {
    "Solve",
    "Explain",
    "CreateGraph",
    "Variant",
    "Solution",
    "Check",
}

FEATURE_ALIAS_MAP = {
    "solve": "Solve",
    "solving": "Solve",
    "explain": "Explain",
    "explanation": "Explain",
    "creategraph": "CreateGraph",
    "graph": "CreateGraph",
    "variant": "Variant",
    "variation": "Variant",
    "solution": "Solution",
    "answer": "Solution",
    "check": "Check",
    "verify": "Check",
}

SOLUTION_KEYWORDS = (
    "해설지",
    "해설지 생성",
    "해설지 생성하기",
    "해설지 pdf",
    "pdf로 생성",
    "pdf로 만들어",
    "pdf로 만들어줘",
    "pdf로 저장",
    "pdf로 저장해",
    "pdf로 저장해줘",
)
SOLVE_KEYWORDS = ("해설해줘", "풀이해줘", "설명해줘", "풀어줘")
NEXT_PROBLEM_KEYWORDS = (
    "다음 문제",
    "다음문제",
    "그다음",
    "이어",
    "계속",
    "next",
)
NEXT_CONFIRM_WORDS = {"응", "네", "예", "ㅇㅇ", "그래", "좋아", "yes"}


def _has_solution_intent(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    has_explicit = any(keyword.lower() in lowered for keyword in SOLUTION_KEYWORDS)
    if _has_solve_intent(text) and not has_explicit:
        return False
    return has_explicit


def _has_solve_intent(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in SOLVE_KEYWORDS)


def _is_next_problem_request(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    normalized = stripped.replace(" ", "").lower()
    if stripped in NEXT_CONFIRM_WORDS or normalized in {
        w.lower() for w in NEXT_CONFIRM_WORDS
    }:
        return True
    return any(
        keyword.replace(" ", "").lower() in normalized
        for keyword in NEXT_PROBLEM_KEYWORDS
    )


def _order_problem_inventory(items: list[dict]) -> list[dict]:
    """문제 번호가 있으면 번호 오름차순으로 정렬하고, 번호 없는 항목은 뒤로 보낸다."""
    try:
        return sorted(
            items,
            key=lambda item: (
                not isinstance(item, dict) or item.get("number") is None,
                int(item.get("number"))
                if isinstance(item, dict) and item.get("number") is not None
                else 10**9,
            ),
        )
    except Exception:
        # 정렬 실패 시 원본 순서를 유지해 하위 호환성을 보장한다.
        return items


def _normalize_feature_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    slug = "".join(ch for ch in raw if ch.isalnum()).lower()
    canonical = FEATURE_ALIAS_MAP.get(slug)
    if canonical in FEATURE_ACTIONS:
        return canonical
    return None


def _extract_chosen_features(state: AgentState) -> List[str]:
    selections = state.get("chosen_features") or []
    normalized: List[str] = []
    for item in selections:
        canonical = _normalize_feature_name(item)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _collect_user_context(state: AgentState) -> tuple[str, str, str, str]:
    """사용자 컨텍스트를 수집합니다.

    Returns:
        tuple: (latest_question, ocr_full_text, combined, conversation_context)
        - latest_question: 가장 최근 사용자 질문
        - ocr_full_text: OCR 추출 텍스트
        - combined: latest_question + OCR 결합
        - conversation_context: 이전 대화 맥락 (멀티턴 지원)
    """
    from agents.workflows.utils import get_conversation_summary

    messages = state.get("messages") or []
    latest_question = ""
    if messages:
        last_message: BaseMessage = messages[-1]
        content = getattr(last_message, "content", "")
        latest_question = content.strip() if isinstance(content, str) else ""

    ocr_full_text = extract_ocr_text(state)

    if latest_question and ocr_full_text:
        combined = f"{latest_question}\n\n[OCR]\n{ocr_full_text}"
    elif ocr_full_text:
        combined = ocr_full_text
    else:
        combined = latest_question

    # 이전 대화 맥락 수집 (checkpointer가 저장한 히스토리 활용)
    conversation_context = ""
    if len(messages) > 1:
        conversation_context = get_conversation_summary(state, max_chars=1500)

    return latest_question, ocr_full_text, combined, conversation_context


def _is_complex_intent(question: str) -> bool:
    if not question:
        return False
    classifier = get_model(OpenRouterModelName.GPT_5_MINI)
    classifier = classifier.with_config(tags=["skip_stream"])
    prompt = [
        SystemMessage(content=COMPLEXITY_SYSTEM_PROMPT),
        HumanMessage(content=build_complexity_user_prompt(question)),
    ]
    try:
        verdict = classifier.invoke(prompt)
        normalized = (getattr(verdict, "content", "") or "").strip().upper()
        if normalized.startswith("MULTI"):
            return True
        if normalized.startswith("SINGLE"):
            return False
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"---ROUTER: COMPLEXITY CLASSIFIER ERROR {exc!r}---")
    return False


def _infer_primary_feature(question: str) -> Optional[str]:
    if not question:
        return None
    if _has_solution_intent(question):
        return "Solution"
    if _has_solve_intent(question):
        return "Solve"
    allowed = ", ".join(sorted(FEATURE_ACTIONS))
    system_prompt = build_primary_feature_system_prompt(allowed)
    human_prompt = build_primary_feature_user_prompt(question)
    model = get_model(OpenRouterModelName.GPT_5_MINI)
    model = model.with_config(tags=["skip_stream"])
    try:
        resp = model.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        )
        candidate = (getattr(resp, "content", "") or "").strip()
        return _normalize_feature_name(candidate)
    except Exception as exc:  # pragma: no cover
        print(f"---ROUTER: FEATURE CLASSIFIER ERROR {exc!r}---")
        return None


def _generate_plan_with_model(
    question: str, hints: Optional[List[str]] = None
) -> List[str]:
    hints = hints or []
    if not question and not hints:
        return []

    allowed = ", ".join(sorted(FEATURE_ACTIONS))
    sections: List[str] = []
    if question:
        sections.append(build_planner_question_section(question))
    if hints:
        sections.append(build_planner_hints_section(", ".join(hints)))
    sections.append(build_planner_output_instruction(allowed))

    model = get_model(OpenRouterModelName.GPT_5_1_CODEX_MINI)
    model = model.with_config(tags=["skip_stream"])
    try:
        ai_message = model.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(sections)),
            ]
        )
        raw_content = getattr(ai_message, "content", "")
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        parsed = json.loads(raw_content)
        candidate_steps = parsed.get("plan") if isinstance(parsed, dict) else None
    except Exception as exc:  # pragma: no cover - best effort parsing
        print(f"---ROUTER: PLAN PARSING ERROR {exc!r}---")
        candidate_steps = None

    plan: List[str] = []
    for hint in hints:
        if hint in FEATURE_ACTIONS and hint not in plan:
            plan.append(hint)

    if isinstance(candidate_steps, list):
        for step in candidate_steps:
            canonical = _normalize_feature_name(step)
            if canonical and canonical not in plan:
                plan.append(canonical)
    return plan


def intent(state: AgentState) -> AgentState:
    """사용자 질문의 의도를 파악하고, RAG 호출 필요 여부 등을 결정합니다.
    어떤 경우든 router 그래프는 여기서 종료되고, maingraph가 다음을 결정합니다.

    checkpointer가 저장한 이전 대화 기록을 활용하여 맥락을 유지합니다.
    난이도 분류는 각 Feature 서브그래프에서 수행합니다.
    """
    print("---ROUTER: INTENT DETECTION---")
    latest_question, ocr_full_text, combined_question, _ = _collect_user_context(state)

    # input_files가 바뀐 경우 이전 인벤토리를 폐기한다.
    current_fingerprint = input_files_fingerprint(state)
    last_fingerprint = state.get("last_input_hash") or ""
    if current_fingerprint and current_fingerprint != last_fingerprint:
        state["last_input_hash"] = current_fingerprint
        state.pop("problems", None)
        state.pop("current_problem_index", None)

    # 인벤토리가 없을 때만 1회 생성한다.
    problems = state.get("problems")
    if not problems:
        inventory = extract_problem_inventory(state)
        if inventory:
            inventory = _order_problem_inventory(inventory)
            state["problems"] = inventory
            state["current_problem_index"] = 0
            problems = inventory

    # 사용자가 "다음" 요청 시, 인덱스만 올려 Solve로 빠르게 라우팅한다.
    if _is_next_problem_request(latest_question):
        indexed_problems = state.get("problems") or []
        if isinstance(indexed_problems, list):
            current_index = int(state.get("current_problem_index", 0) or 0)
            if 0 <= current_index + 1 < len(indexed_problems):
                state["current_problem_index"] = current_index + 1
                state["plan"] = ["Solve"]
                state["simple_response"] = False
                state["prev_action"] = "Intent"
                return state

    latest_question, ocr_full_text, combined_question, conversation_context = (
        _collect_user_context(state)
    )
    chosen = _extract_chosen_features(state)

    if "Solution" in chosen or _has_solution_intent(combined_question):
        state["simple_response"] = False
        state["prev_action"] = "Intent"
        return state

    if combined_question:
        classifier = get_model(OpenRouterModelName.GPT_5_MINI)
        classifier = classifier.with_config(tags=["skip_stream"])

        # 대화 맥락이 있으면 포함하여 더 정확한 분류
        context_info = ""
        if conversation_context:
            context_info = build_stem_context_section(conversation_context)
        prompt_messages = [
            SystemMessage(content=STEM_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=build_stem_user_prompt(context_info, combined_question)
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


def intent_route(state: AgentState) -> Literal["Planner", "Executor", "__end__"]:
    """RAG 검색 결과와 사용자 의도를 종합하여 단일/복합 의도를 결정합니다.
    - simple_response인 경우: 즉시 종료 (maingraph에서 Simple_response로 라우팅)
    - 복합 의도: 실행 계획 수립을 위해 Planner로 이동
    - 단일 의도: 바로 실행을 위해 Executor로 이동
    """
    print("---ROUTER: INTENT ROUTING---")

    # simple_response가 True면 Feature 실행 없이 바로 종료
    # maingraph의 route_to_feature에서 Simple_response로 라우팅됨
    if state.get("simple_response") is True:
        print("---ROUTER: SIMPLE RESPONSE DETECTED, SKIPPING FEATURES---")
        return "__end__"

    existing_plan = [
        step for step in (state.get("plan") or []) if step in FEATURE_ACTIONS
    ]
    if existing_plan:
        state["plan"] = existing_plan
        return "Executor"

    chosen = _extract_chosen_features(state)
    if len(chosen) == 1:
        state["plan"] = chosen.copy()
        return "Executor"
    if len(chosen) > 1:
        return "Planner"

    _, _, combined_question, _ = _collect_user_context(state)
    if _has_solution_intent(combined_question):
        state["plan"] = ["Solution"]
        return "Executor"
    if _has_solve_intent(combined_question):
        state["plan"] = ["Solve"]
        return "Executor"
    requires_planner = _is_complex_intent(combined_question)
    if requires_planner:
        return "Planner"

    primary_feature = _infer_primary_feature(combined_question) or "Solve"
    state["plan"] = [primary_feature]
    return "Executor"


def planner(state: AgentState) -> AgentState:
    """복합 의도에 대한 실행 계획을 수립합니다."""
    print("---ROUTER: PLANNING---")
    _, _, combined_question, _ = _collect_user_context(state)
    hints = _extract_chosen_features(state)
    plan = _generate_plan_with_model(combined_question, hints)

    if not plan and hints:
        plan = hints.copy()
    if not plan:
        fallback = _infer_primary_feature(combined_question) or "Solve"
        plan = [fallback]

    print(f"---ROUTER: PLAN RESULT {plan}---")
    state["plan"] = plan
    state["retry_count"] = 0
    state.pop("current_step", None)
    state["prev_action"] = "Planner"
    return state


def executor(state: AgentState) -> AgentState:
    """수립된 계획의 다음 단계를 실행 준비합니다."""
    print("---ROUTER: EXECUTING STEP---")
    plan = [step for step in (state.get("plan") or []) if step in FEATURE_ACTIONS]

    if env_truthy(FORCE_RETRY_EXECUTOR_TO_REVIEW_ENV, default="false"):
        # Retry flow 테스트 시 Feature 노드 대신 Review로 직접 라우팅한다.
        state["current_step"] = "Review"
        state["plan"] = plan
        state["prev_action"] = "Executor"
        return state

    if not plan:
        _, _, combined_question, _ = _collect_user_context(state)
        hints = _extract_chosen_features(state)
        fallback = hints[0] if hints else _infer_primary_feature(combined_question)
        plan = [fallback or "Solve"]

    current_step = plan.pop(0)
    state["current_step"] = current_step
    state["plan"] = plan
    state["prev_action"] = "Executor"
    return state


def retry_counter(
    state: AgentState, config: RunnableConfig | None = None
) -> Literal["Executor", "__end__"]:
    """재시도 횟수를 확인하고, 다음 단계를 결정합니다.
    - 재시도 가능: Executor로 돌아가 다시 실행
    - 재시도 불가: Fallback 응답을 위해 그래프 종료
    """
    print("---ROUTER: RETRY COUNTER---")
    before = int(state.get("retry_count", 0) or 0)
    retry_trace_log(
        event="before_retry_increment",
        retry_count=before,
        node="RetryCounter",
        state=state,
        config=config,
        level="info",
    )

    if before >= MAX_RETRIES:
        state["retry_limit_exceeded"] = True
        retry_trace_log(
            event="retry_limit_reached",
            retry_count=before,
            node="RetryCounter",
            state=state,
            config=config,
            level="warning",
        )
        print(f"---ROUTER: RETRY LIMIT EXCEEDED ({before}/{MAX_RETRIES})---")
        return "__end__"

    retries = before + 1
    state["retry_count"] = retries
    state.pop("retry_limit_exceeded", None)
    print(f"---ROUTER: RETRYING ({retries}/{MAX_RETRIES})---")
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

    return "Executor"


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
        "__end__": END,
    },
)

# 3. 계획 수립 및 실행
builder.add_edge("Planner", "Executor")
builder.add_edge("Executor", END)


# 4. 재시도(Retry) 진입점
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
