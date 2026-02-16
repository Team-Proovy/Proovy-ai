"""Shared workflow-level utilities for LangGraph agents.

이 모듈은 여러 step4_features 서브그래프에서 공통으로 사용하는
텍스트 추출, JSON 파싱, LLM 호출 유틸리티를 모아둔 곳입니다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from core.llm import get_model
from schema.models import OpenRouterModelName
from agents.prompts.difficulty_prompts import (
    DIFFICULTY_CLASSIFIER_SYSTEM_PROMPT,
    DIFFICULTY_MODEL_MAP,
    build_difficulty_classifier_user_prompt,
    get_model_for_difficulty,
)


def message_to_text(message: BaseMessage) -> str:
    """LangChain BaseMessage에서 순수 텍스트 콘텐츠를 추출한다."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text") or chunk.get("data") or ""))
            else:
                parts.append(str(chunk))
        return "\n".join([p for p in parts if p]).strip()
    return str(content)


def get_conversation_history(
    state: AgentState,
    *,
    max_turns: int = 5,
    max_chars_per_message: int = 1000,
    include_system: bool = False,
) -> List[BaseMessage]:
    """LLM에 전달할 대화 히스토리를 LangChain 메시지 리스트로 반환한다.

    checkpointer가 저장한 이전 대화 기록을 활용하여
    멀티턴 대화의 맥락을 LLM에 전달할 수 있도록 합니다.

    Args:
        state: 현재 AgentState
        max_turns: 포함할 최대 대화 턴 수 (user+assistant = 1턴)
        max_chars_per_message: 메시지당 최대 문자 수
        include_system: SystemMessage도 포함할지 여부

    Returns:
        LangChain 메시지 리스트 (HumanMessage, AIMessage 등)
    """
    messages = state.get("messages") or []
    result: List[BaseMessage] = []
    turns_collected = 0
    last_type = None

    for message in reversed(messages):
        msg_type = getattr(message, "type", "")

        # 시스템 메시지는 선택적으로 포함
        if msg_type == "system":
            if include_system:
                result.append(message)
            continue

        # tool 메시지는 건너뜀
        if msg_type == "tool":
            continue

        # 유효한 타입만 처리
        if msg_type not in {"human", "user", "ai", "assistant"}:
            continue

        # 턴 카운트: user -> assistant 전환 시 1턴
        is_user = msg_type in {"human", "user"}
        if last_type is not None:
            if is_user and last_type in {"ai", "assistant"}:
                turns_collected += 1

        if turns_collected >= max_turns:
            break

        # 메시지 텍스트 추출 및 truncate
        text = message_to_text(message).strip()
        if not text:
            continue

        if len(text) > max_chars_per_message:
            text = text[:max_chars_per_message] + "..."

        # 적절한 메시지 타입으로 변환
        if is_user:
            result.append(HumanMessage(content=text))
        else:
            result.append(AIMessage(content=text))

        last_type = msg_type

    result.reverse()
    return result


def get_conversation_summary(state: AgentState, *, max_chars: int = 2000) -> str:
    """대화 히스토리를 간략한 요약 문자열로 반환한다.

    시스템 프롬프트에 대화 맥락을 포함시킬 때 유용합니다.
    """
    messages = state.get("messages") or []
    summary_parts: List[str] = []
    total_chars = 0

    for message in messages:
        msg_type = getattr(message, "type", "")
        if msg_type in {"system", "tool"}:
            continue

        text = message_to_text(message).strip()
        if not text:
            continue

        role = "사용자" if msg_type in {"human", "user"} else "AI"
        # 긴 메시지는 첫 200자만
        preview = text[:200] + "..." if len(text) > 200 else text
        entry = f"{role}: {preview}"

        if total_chars + len(entry) > max_chars:
            break

        summary_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(summary_parts)


def recent_user_context(
    state: AgentState,
    *,
    max_messages: int = 3,
    include_assistant: bool = False,
    max_assistant_chars: int = 500,
) -> str:
    """최근 사용자(human/user) 메시지 몇 개를 이어붙여 컨텍스트 문자열로 만든다.

    Args:
        state: 현재 AgentState
        max_messages: 수집할 최대 메시지 수
        include_assistant: True면 AI 응답도 포함하여 대화 맥락 유지
        max_assistant_chars: AI 응답 포함 시 최대 문자 수 (요약용)
    """
    messages = state.get("messages") or []
    chunks: List[str] = []
    collected = 0

    for message in reversed(messages):
        msg_type = getattr(message, "type", "")
        if msg_type in {"human", "user"}:
            text = message_to_text(message).strip()
            if text:
                chunks.append(f"[사용자]: {text}")
                collected += 1
        elif include_assistant and msg_type in {"ai", "assistant"}:
            text = message_to_text(message).strip()
            if text:
                # AI 응답은 길 수 있으므로 요약
                truncated = text[:max_assistant_chars]
                if len(text) > max_assistant_chars:
                    truncated += "..."
                chunks.append(f"[AI]: {truncated}")
                collected += 1

        if collected >= max_messages:
            break

    chunks.reverse()
    return "\n\n".join(chunks).strip()


def extract_ocr_text(state: AgentState) -> str:
    """Preprocessing 단계에서 저장한 OCR 블록을 단일 문자열로 병합한다."""
    file_processing = state.get("file_processing")
    if not file_processing:
        return ""
    if isinstance(file_processing, dict):
        ocr_data = file_processing.get("ocr_blocks")
    else:
        ocr_data = getattr(file_processing, "ocr_blocks", None)
    if not ocr_data:
        return ""

    pages = None
    if isinstance(ocr_data, dict):
        pages = ocr_data.get("pages")
    elif isinstance(ocr_data, list):
        pages = ocr_data
    else:
        pages = getattr(ocr_data, "pages", None)

    if not isinstance(pages, list):
        return str(ocr_data).strip()

    def _block_text(block: Any) -> str:
        if isinstance(block, dict):
            text = block.get("text")
            latex = block.get("latex")
        else:
            text = getattr(block, "text", None)
            latex = getattr(block, "latex", None)
        parts: List[str] = []
        if text:
            parts.append(str(text).strip())
        if latex:
            latex_value = str(latex).strip()
            if latex_value and latex_value not in parts:
                parts.append(latex_value)
        if parts:
            return "\n".join(parts).strip()
        if isinstance(block, dict):
            return ""
        return str(block).strip()

    page_texts: List[str] = []
    for page in pages:
        if isinstance(page, dict):
            blocks = page.get("blocks") or []
        else:
            blocks = getattr(page, "blocks", None) or []
        if not isinstance(blocks, list):
            blocks = [blocks]
        block_texts: List[str] = []
        for block in blocks:
            text = _block_text(block)
            if text:
                block_texts.append(text)
        if block_texts:
            page_texts.append("\n".join(block_texts))

    return "\n\n".join(page_texts).strip()


def safe_json_loads(raw: str) -> Dict[str, Any]:
    """LLM 응답처럼 난잡할 수 있는 문자열에서 JSON 딕셔너리를 최대한 안전하게 파싱한다."""
    cleaned = raw.strip()
    if not cleaned:
        return {}
    if cleaned.startswith("```"):
        segments = []
        for part in cleaned.split("```"):
            part = part.strip()
            if not part or part.lower().startswith("json"):
                continue
            segments.append(part)
        cleaned = "\n".join(segments).strip() or cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if start != -1 and end != -1 else cleaned
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return json.loads(candidate.replace("'", '"'))
        except json.JSONDecodeError:
            return {}


def ensure_str_list(value: Any) -> List[str]:
    """단일 값 또는 리스트를 정제된 문자열 리스트로 통일한다."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def classify_difficulty(question: str) -> str:
    """문제의 난이도를 LLM으로 분류합니다.

    각 Feature 서브그래프에서 호출하여 난이도를 결정하고,
    그에 맞는 모델을 선택하는 데 사용합니다.

    난이도별 사용 모델:
    - easy: Gemini 2.5 Flash
    - medium: Gemini 3 Flash
    - hard: Gemini 3 Pro

    Args:
        question: 문제/질문 텍스트

    Returns:
        난이도 (easy, medium, hard)
    """
    if not question:
        return "easy"

    classifier = get_model(OpenRouterModelName.GEMINI_25_FLASH)
    classifier = classifier.with_config(tags=["skip_stream"])

    user_prompt = build_difficulty_classifier_user_prompt(question[:2000])

    prompt = [
        SystemMessage(content=DIFFICULTY_CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        result = classifier.invoke(prompt)
        verdict = (getattr(result, "content", "") or "").strip().upper()

        if verdict.startswith("HARD"):
            return "hard"
        elif verdict.startswith("MEDIUM"):
            return "medium"
        else:
            return "easy"
    except Exception as exc:
        print(f"---DIFFICULTY CLASSIFIER ERROR {exc!r}---")
        return "easy"


def set_difficulty_in_state(state: AgentState, difficulty: str) -> None:
    """분류된 난이도를 credit_state와 router_state에 저장합니다.

    Args:
        state: AgentState
        difficulty: 난이도 (easy, medium, hard)
    """
    # credit_state 업데이트
    credit_state = state.get("credit_state")
    if credit_state:
        if isinstance(credit_state, dict):
            credit_state["difficulty"] = difficulty
        else:
            credit_state.difficulty = difficulty
        state["credit_state"] = credit_state
    else:
        from agents.state import CreditState
        state["credit_state"] = CreditState(difficulty=difficulty)

    # router_state에도 저장 (하위 호환성)
    router_state = state.get("router_state") or {}
    router_state["difficulty"] = difficulty
    state["router_state"] = router_state


def _normalize_difficulty(value: Any) -> str:
    difficulty = str(value).strip().lower()
    return difficulty if difficulty in DIFFICULTY_MODEL_MAP else "easy"


def get_difficulty_from_state(state: AgentState) -> str:
    """state에서 난이도를 추출합니다.

    Returns:
        난이도 문자열 (easy, medium, hard). 기본값은 "easy"
    """
    # credit_state에서 확인
    credit_state = state.get("credit_state")
    if credit_state:
        if isinstance(credit_state, dict):
            return _normalize_difficulty(credit_state.get("difficulty", "easy"))
        return _normalize_difficulty(getattr(credit_state, "difficulty", "easy"))

    # router_state에서 확인
    router_state = state.get("router_state")
    if router_state:
        if isinstance(router_state, dict):
            return _normalize_difficulty(router_state.get("difficulty", "easy"))
        return _normalize_difficulty(getattr(router_state, "difficulty", "easy"))

    return "easy"


def get_model_name_for_state(state: AgentState) -> OpenRouterModelName:
    """state의 난이도에 맞는 LLM 모델을 반환합니다.

    난이도별 모델:
    - easy: Gemini 2.5 Flash (빠르고 저렴)
    - medium: Gemini 3 Flash (대부분의 문제)
    - hard: Gemini 3 Pro (복잡한 문제만)
    """
    difficulty = get_difficulty_from_state(state)
    return get_model_for_difficulty(difficulty)


def call_model(
    model_name: OpenRouterModelName,
    system_prompt: str,
    user_prompt: str,
    *,
    tags: list[str] | None = None,
):
    """기본적인 system+human 프롬프트 패턴으로 OpenRouter 모델을 호출한다.

    기본값으로 `tags=["skip_stream"]`를 설정해, 이 유틸을 사용하는 대부분의 노드에서는
    /stream 토큰 스트리밍 대상에서 제외되도록 한다. 토큰을 스트리밍해야 하는 노드는
    tags=[] 또는 원하는 태그 목록을 명시적으로 전달한다.
    """
    if tags is None:
        tags = ["skip_stream"]

    base_model = get_model(model_name)
    model = base_model.with_config(tags=tags) if tags else base_model

    ai_message = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = getattr(ai_message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            (chunk.get("text") if isinstance(chunk, dict) else str(chunk))
            for chunk in content
        )
    return str(content)


def call_model_by_difficulty(
    state: AgentState,
    system_prompt: str,
    user_prompt: str,
    *,
    tags: list[str] | None = None,
):
    """state의 난이도에 맞는 모델로 LLM을 호출합니다.

    난이도별 모델:
    - easy: Gemini 2.5 Flash
    - medium: Gemini 3 Flash
    - hard: Gemini 3 Pro

    Args:
        state: AgentState (난이도 정보 포함)
        system_prompt: 시스템 프롬프트
        user_prompt: 사용자 프롬프트
        tags: LangGraph 태그 (기본값: ["skip_stream"])

    Returns:
        LLM 응답 텍스트
    """
    model_name = get_model_name_for_state(state)
    difficulty = get_difficulty_from_state(state)
    print(f"→ Using model for difficulty '{difficulty}': {model_name}")
    return call_model(model_name, system_prompt, user_prompt, tags=tags)


# =============================================================================
# 미들웨어 기반 비용 추적 (Cost Tracking Middleware)
# =============================================================================

# 난이도별 토큰당 비용 (크레딧 단위)
COST_PER_1K_TOKENS = {
    "easy": 0.5,    # Gemini 2.5 Flash - 저렴
    "medium": 1.0,  # Gemini 3 Flash - 중간
    "hard": 3.0,    # Gemini 3 Pro - 비쌈
}


class InsufficientCreditError(Exception):
    """크레딧 부족 시 발생하는 예외"""
    def __init__(self, balance: float, total_cost: float, node_name: str = ""):
        self.balance = balance
        self.total_cost = total_cost
        self.node_name = node_name
        self.message = f"크레딧 부족: 잔액 {balance}, 누적 비용 {total_cost}"
        if node_name:
            self.message += f" (노드: {node_name})"
        super().__init__(self.message)


def calculate_token_cost(response_metadata: Dict[str, Any], difficulty: str = "easy") -> float:
    """LLM 응답의 토큰 사용량을 기반으로 비용을 계산합니다.

    Args:
        response_metadata: LLM 응답의 메타데이터 (usage_metadata 포함)
        difficulty: 문제 난이도 (비용 단가 결정)

    Returns:
        계산된 비용 (크레딧 단위)
    """
    usage = response_metadata.get("usage_metadata") or response_metadata.get("token_usage") or {}

    # 토큰 사용량 추출
    total_tokens = usage.get("total_tokens", 0)
    if total_tokens == 0:
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

    # 비용 계산
    cost_per_1k = COST_PER_1K_TOKENS.get(difficulty, 1.0)
    cost = (total_tokens / 1000) * cost_per_1k

    return round(cost, 4)


def check_credit_before_node(state: AgentState, node_name: str = "") -> None:
    """노드 실행 전 크레딧 잔액을 체크합니다.

    Args:
        state: AgentState
        node_name: 실행하려는 노드 이름

    Raises:
        InsufficientCreditError: 잔액 부족 시
    """
    credit_state = state.get("credit_state")
    if not credit_state:
        return  # 크레딧 상태 없으면 체크 스킵

    if isinstance(credit_state, dict):
        balance = credit_state.get("balance", 0)
        total_cost = credit_state.get("total_cost", 0)
    else:
        balance = getattr(credit_state, "balance", 0)
        total_cost = getattr(credit_state, "total_cost", 0)

    if total_cost >= balance:
        raise InsufficientCreditError(balance, total_cost, node_name)


def update_credit_after_node(
    state: AgentState,
    node_name: str,
    response_metadata: Dict[str, Any] | None = None,
    fixed_cost: float | None = None,
) -> Dict[str, Any]:
    """노드 실행 후 크레딧 비용을 업데이트합니다.

    Args:
        state: AgentState
        node_name: 실행된 노드 이름
        response_metadata: LLM 응답 메타데이터 (토큰 기반 비용 계산용)
        fixed_cost: 고정 비용 (토큰 계산 대신 사용)

    Returns:
        업데이트된 credit_state dict
    """
    credit_state = state.get("credit_state") or {}
    if not isinstance(credit_state, dict):
        credit_state = credit_state.model_dump() if hasattr(credit_state, 'model_dump') else {}

    difficulty = credit_state.get("difficulty", "easy")

    # 비용 계산
    if fixed_cost is not None:
        cost = fixed_cost
    elif response_metadata:
        cost = calculate_token_cost(response_metadata, difficulty)
    else:
        cost = 0.0

    # 비용 누적
    credit_state["total_cost"] = credit_state.get("total_cost", 0) + cost

    # 노드별 비용 기록
    cost_per_node = credit_state.get("cost_per_node", {})
    cost_per_node[node_name] = cost_per_node.get(node_name, 0) + cost
    credit_state["cost_per_node"] = cost_per_node

    # 잔액 체크
    balance = credit_state.get("balance", 0)
    if credit_state["total_cost"] >= balance:
        credit_state["insufficient"] = True
        credit_state["stopped_at_feature"] = node_name

    print(f"→ Credit updated: {node_name} cost={cost}, total={credit_state['total_cost']}/{balance}")

    return credit_state


def credit_guard_wrapper(node_func):
    """크레딧 가드 미들웨어 데코레이터.

    노드 실행 전 잔액 체크 + 실행 후 비용 업데이트를 자동화합니다.

    사용법:
        @credit_guard_wrapper
        def my_node(state: AgentState) -> AgentState:
            ...
    """
    def wrapper(state: AgentState) -> AgentState:
        node_name = node_func.__name__

        # 1. [BEFORE] 실행 전 체크
        try:
            check_credit_before_node(state, node_name)
        except InsufficientCreditError as e:
            print(f"→ Credit guard blocked: {e.message}")
            # 크레딧 부족 상태 설정
            credit_state = state.get("credit_state") or {}
            if isinstance(credit_state, dict):
                credit_state["insufficient"] = True
                credit_state["stopped_at_feature"] = node_name
                state["credit_state"] = credit_state
            return state

        # 2. 노드 실행
        result = node_func(state)

        # 3. [AFTER] 실행 후 비용 업데이트
        # 메시지에서 usage 정보 추출
        messages = result.get("messages") or []
        response_metadata = {}
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "response_metadata"):
                response_metadata = last_msg.response_metadata or {}
            elif hasattr(last_msg, "usage_metadata"):
                response_metadata = {"usage_metadata": last_msg.usage_metadata}

        updated_credit = update_credit_after_node(result, node_name, response_metadata)
        result["credit_state"] = updated_credit

        return result

    # 함수 메타데이터 보존
    wrapper.__name__ = node_func.__name__
    wrapper.__doc__ = node_func.__doc__
    return wrapper


def check_credit_sufficient(state: AgentState) -> bool:
    """Conditional Edge용: 크레딧이 충분한지 확인합니다.

    Returns:
        True: 크레딧 충분
        False: 크레딧 부족
    """
    credit_state = state.get("credit_state")
    if not credit_state:
        return True

    if isinstance(credit_state, dict):
        balance = credit_state.get("balance", float("inf"))
        total_cost = credit_state.get("total_cost", 0)
        insufficient = credit_state.get("insufficient", False)
    else:
        balance = getattr(credit_state, "balance", float("inf"))
        total_cost = getattr(credit_state, "total_cost", 0)
        insufficient = getattr(credit_state, "insufficient", False)

    if insufficient:
        return False

    return total_cost < balance
