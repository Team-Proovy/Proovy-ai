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
    """Preprocessing 단계에서 저장한 OCR 텍스트를 단일 문자열로 병합한다."""
    file_processing = state.get("file_processing")
    if not file_processing or not getattr(file_processing, "ocr_text", None):
        return ""
    ocr_data = getattr(file_processing, "ocr_text") or {}
    if isinstance(ocr_data, dict):
        if ocr_data.get("full_text"):
            return str(ocr_data["full_text"]).strip()
        pages = ocr_data.get("pages")
        if isinstance(pages, list):
            merged = "\n".join(str(page) for page in pages if page)
            return merged.strip()
    return str(ocr_data)


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
):
    """기본적인 system+human 프롬프트 패턴으로 OpenRouter 모델을 호출한다."""
    model = get_model(model_name)
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
