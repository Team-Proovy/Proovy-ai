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
import re
from typing import Literal, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import AgentState
from agents.workflows.utils import extract_ocr_text
from core.llm import get_model
from schema.models import OpenRouterModelName

MAX_RETRIES = 2


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
SOLVE_ALL_KEYWORDS = ("다 풀어줘", "전부", "모두", "전부 해설", "모두 풀어줘", "다 해설", "한꺼번에")


def _detect_solve_all_intent(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in SOLVE_ALL_KEYWORDS)


def _detect_target_number(text: str) -> Optional[int]:
    """사용자 메시지에서 'n번' 혹은 'n번 문제'와 같은 타겟 번호를 추출합니다."""
    if not text:
        return None
    # '2번', '2번 문제', '2번만', 'Q2' 등 패턴 매칭
    match = re.search(r"(\d+)\s*(?:번|번\s*문제|번만)", text)
    if match:
        return int(match.group(1))
    
    match_q = re.search(r"(?:Q|No|#)\s*(\d+)", text, re.IGNORECASE)
    if match_q:
        return int(match_q.group(1))
    return None


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


def _collect_user_context(state: AgentState) -> tuple[str, str, str]:
    messages = state.get("messages") or []
    latest_question = ""
    if messages:
        last_message: BaseMessage = messages[-1]
        content = getattr(last_message, "content", "")
        
        # [치명적 에러 방지] content가 list(Multimodal)일 경우 대응
        if isinstance(content, str):
            latest_question = content.strip()
        elif isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            latest_question = "\n".join(text_parts).strip()
        else:
            latest_question = str(content or "").strip()

    ocr_full_text = extract_ocr_text(state)

    if latest_question and ocr_full_text:
        combined = f"{latest_question}\n\n[OCR]\n{ocr_full_text}"
    elif ocr_full_text:
        combined = ocr_full_text
    else:
        combined = latest_question
    return latest_question, ocr_full_text, combined


def _is_complex_intent(question: str) -> bool:
    if not question:
        return False
    classifier = get_model(OpenRouterModelName.GPT_5_MINI)
    classifier = classifier.with_config(tags=["skip_stream"])
    system_prompt = (
        "You are an intent assessor. "
        "Return 'MULTI' if the request needs multiple distinct reasoning steps "
        "(e.g., solve then explain, or create variants after solving). "
        "Return 'SINGLE' if one feature is enough."
    )
    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question:\n{question}\n\nAnswer SINGLE or MULTI."),
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
    system_prompt = (
        "You map a math-related request to the most suitable feature. "
        f"Choose exactly one from [{allowed}]."
    )
    human_prompt = f"Question:\n{question}\n\nReturn only the chosen feature name."
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
    system_prompt = (
        "You are a meticulous planner for a math tutoring agent. "
        "Break the task into an ordered list of features chosen from the allowed set. "
        "Return valid JSON so that downstream code can parse it."
    )
    sections: List[str] = []
    if question:
        sections.append(f"Question:\n{question}")
    if hints:
        sections.append(
            "User explicitly selected features (respect this order when reasonable): "
            + ", ".join(hints)
        )
    sections.append(
        'Respond with JSON like {"plan": ["Solve", "Explain"]} where each item '
        f"belongs to [{allowed}] and there are no duplicates."
    )

    model = get_model(OpenRouterModelName.GPT_5_1_CODEX_MINI)
    model = model.with_config(tags=["skip_stream"])
    try:
        ai_message = model.invoke(
            [
                SystemMessage(content=system_prompt),
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
    """
    print("---ROUTER: INTENT DETECTION---")
    latest_question, ocr_full_text, combined_question = _collect_user_context(state)

    # [개선] 1. 문제 분할의 멱등성(Idempotency) 보장
    # 이미 충분한 문제(chunks)가 확보되었다면, 새로운 맥락이 아닌 한 재분할을 피합니다.
    from agents.workflows.subgraphs.step4_features.solution.problem_utils import split_text_into_problems
    
    current_chunks = state.get("solution_chunks") or []
    is_new_context = not current_chunks # 처음 시작하거나 chunks가 비어있는 경우
    
    if is_new_context and combined_question:
        # [최종 개선] 입력 텍스트의 앞뒤 공백을 제거하여 마지막 유령 청크 생성을 원천 차단합니다.
        new_chunks = split_text_into_problems(combined_question.strip())
        if len(new_chunks) > 0:
            state["solution_chunks"] = list(new_chunks)
            print(f"---ROUTER: INITIAL PROBLEM SPLIT ({len(new_chunks)} problems)---")

    chosen = _extract_chosen_features(state)
    
    target_num = _detect_target_number(combined_question)
    if target_num:
        state["target_problem_number"] = target_num
        print(f"---ROUTER: TARGET PROBLEM DETECTED: {target_num}---")
    
    print(f"---ROUTER DEBUG target_num={target_num} chunks_count={len(state.get('solution_chunks') or [])}---")

    if "Solution" in chosen or _has_solution_intent(combined_question):
        state["simple_response"] = False
        state["solve_all"] = _detect_solve_all_intent(combined_question)
        state["prev_action"] = "Intent"
        return state

    if combined_question:
        state["solve_all"] = _detect_solve_all_intent(combined_question)
        classifier = get_model(OpenRouterModelName.GPT_5_MINI)
        classifier = classifier.with_config(tags=["skip_stream"])
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
                    f"{combined_question}\n\n"
                    "Answer with either STEM or NON_STEM."
                )
            ),
        ]
        try:
            ai_message = classifier.invoke(prompt_messages)
            verdict = getattr(ai_message, "content", "")
            normalized = (verdict or "").strip().upper()
            is_stem = normalized.startswith("STEM")
        except Exception as exc: 
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

    latest_question, _, combined_question = _collect_user_context(state)

    current_chunks = state.get("solution_chunks") or []
    target_num = state.get("target_problem_number")
    last_idx = state.get("last_solved_index")

    # [수정] 1. 명시적인 해설지(PDF) 생성 의도를 최우선으로 체크하여 라우팅 블락 방지
    if _has_solution_intent(combined_question):
        state["plan"] = ["Solution"]
        return "Executor"

    # [수정] 2. 특정 번호 타겟팅이 있거나, 명확한 풀이 의도가 있는 경우 Solve로 직행
    # last_idx가 있을 때는 사용자의 답변이 긍정적(다음, 응 등)일 때만 루프를 이어감
    is_solve_request = _has_solve_intent(latest_question) or target_num is not None
    
    # 루프 진행 중 긍정 응답 체크 (latest_question 기준)
    CONTINUE_KEYWORDS = ("다음", "계속", "이어서", "응", "그래", "ok", "yes")
    is_continue_request = last_idx is not None and any(k in latest_question.lower() for k in CONTINUE_KEYWORDS)

    if is_solve_request or is_continue_request:
        state["plan"] = ["Solve"]
        print(f"---ROUTER: SOLVE OR CONTINUE DETECTED -> GO TO EXECUTOR---")
        return "Executor"

    # [개선] 3. 다중 문제 대응 및 라우팅 로직 고도화
    # 문제가 여러 개이고 처음 시작하는 복합 의도일 때만 Planner로 이동
    if len(current_chunks) > 1 or _is_complex_intent(combined_question):
        print(f"---ROUTER: MULTI-PROBLEM OR COMPLEX INTENT -> GO TO PLANNER---")
        return "Planner"

    if _has_solve_intent(combined_question):
        state["plan"] = ["Solve"]
        return "Executor"

    primary_feature = _infer_primary_feature(combined_question) or "Solve"
    state["plan"] = [primary_feature]
    return "Executor"


def planner(state: AgentState) -> AgentState:
    """복합 의도에 대한 실행 계획을 수립합니다."""
    print("---ROUTER: PLANNING---")
    _, _, combined_question = _collect_user_context(state)
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

    if not plan:
        _, _, combined_question = _collect_user_context(state)
        hints = _extract_chosen_features(state)
        fallback = hints[0] if hints else _infer_primary_feature(combined_question)
        plan = [fallback or "Solve"]

    current_step = plan.pop(0)
    state["current_step"] = current_step
    state["plan"] = plan
    state["prev_action"] = "Executor"
    return state


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
