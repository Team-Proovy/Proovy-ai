"""Shared workflow-level utilities for LangGraph agents.

이 모듈은 여러 step4_features 서브그래프에서 공통으로 사용하는
텍스트 추출, JSON 파싱, LLM 호출 유틸리티를 모아둔 곳입니다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from core.llm import get_model
from schema.models import OpenRouterModelName


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


def recent_user_context(state: AgentState, *, max_messages: int = 3) -> str:
    """최근 사용자(human/user) 메시지 몇 개를 이어붙여 컨텍스트 문자열로 만든다."""
    messages = state.get("messages") or []
    user_chunks: List[str] = []
    for message in reversed(messages):
        if getattr(message, "type", "") in {"human", "user"}:
            text = message_to_text(message).strip()
            if text:
                user_chunks.append(text)
        if len(user_chunks) >= max_messages:
            break
    user_chunks.reverse()
    return "\n\n".join(user_chunks).strip()


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
