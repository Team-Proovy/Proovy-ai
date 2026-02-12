"""RAG subgraph.

EmbeddingSearch → RelevanceCheck → RetrievedDocs 흐름의 뼈대입니다.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from time import perf_counter

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from agents.state import AgentState
from agents.workflows.utils import extract_ocr_text, recent_user_context
from agents.workflows.subgraphs.step3_rag import config
from agents.workflows.subgraphs.step3_rag.types import RetrievedDoc, RetrievedDocModel
from rag.retriever import search as retriever_search

logger = logging.getLogger(__name__)


def _build_query_text(state: AgentState) -> Tuple[str, str, str]:
    question = recent_user_context(state)
    ocr_text = extract_ocr_text(state)
    
    # 1. 원본 조합 (LLM 참고용)
    if question and ocr_text:
        combined = f"{question}\n\n[OCR]\n{ocr_text}"
    elif ocr_text:
        combined = ocr_text
    else:
        combined = question

    # 2. 검색용 클린 쿼리 생성 (벡터 검색 노이즈 제거)
    # [OCR] 태그나 "알려줘", "풀어줘" 같은 서술어는 벡터 검색 점수를 낮춥니다.
    search_query = combined
    for noise in ["[OCR]", "알려줘", "풀어줘", "설명해줘", "부탁해", "가르쳐줘"]:
        search_query = search_query.replace(noise, "")
    
    # 불필요한 공백/줄바꿈 정리
    search_query = " ".join(search_query.split()).strip()

    return question, ocr_text, search_query


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_doc(doc: Any, index: int) -> RetrievedDoc:
    if isinstance(doc, dict):
        metadata = doc.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        text = doc.get("text") or doc.get("content") or doc.get("page_content") or ""
        score = _coerce_score(doc.get("score"))
        doc_id = (
            doc.get("id")
            or metadata.get("chunk_id")
            or metadata.get("id")
            or f"doc_{index + 1}"
        )
        title = doc.get("title") or metadata.get("title") or f"Document {index + 1}"
        normalized = {
            "id": str(doc_id),
            "title": str(title),
            "text": str(text),
            "score": score,
            "metadata": metadata,
        }
    else:
        normalized = {
            "id": f"doc_{index + 1}",
            "title": f"Document {index + 1}",
            "text": str(doc),
            "score": None,
            "metadata": {},
        }
    try:
        return RetrievedDocModel(**normalized).model_dump()
    except ValidationError:
        logger.warning("Invalid retrieved doc at index %s", index, exc_info=True)
        fallback = {
            "id": str(normalized.get("id") or f"doc_{index + 1}"),
            "title": str(normalized.get("title") or f"Document {index + 1}"),
            "text": str(normalized.get("text") or ""),
            "score": _coerce_score(normalized.get("score")),
            "metadata": (
                normalized.get("metadata")
                if isinstance(normalized.get("metadata"), dict)
                else {}
            ),
        }
        return RetrievedDocModel(**fallback).model_dump()


def embedding_search(state: AgentState) -> AgentState:
    """유사도 검색 (mock)."""
    logger.info("[RAG] EmbeddingSearch start")
    question, ocr_text, combined = _build_query_text(state)
    tool_outputs = state.setdefault("tool_outputs", {})
    context_texts = [ocr_text] if ocr_text else []
    query = (combined or question or "").strip()
    start = perf_counter()
    try:
        docs = retriever_search(
            query,
            context_texts=context_texts,
            top_k=config.TOP_K,
        )
    except Exception:
        logger.exception("embedding_search failed")
        docs = []
    elapsed_ms = int((perf_counter() - start) * 1000)
    tool_outputs["retrieved_docs"] = docs
    rag_meta = tool_outputs.setdefault("rag_meta", {})
    rag_meta.update(
        {"retrieved_count": len(docs), "embedding_latency_ms": elapsed_ms}
    )
    logger.info("[RAG] EmbeddingSearch done: count=%s elapsed_ms=%s", len(docs), elapsed_ms)
    return state


def relevance_check(state: AgentState) -> AgentState:
    """유사도 threshold 체크."""
    logger.info("[RAG] RelevanceCheck start")
    state["prev_action"] = "RAG"
    state["next_action"] = "IntentRoute"
    tool_outputs = state.setdefault("tool_outputs", {})
    retrieved_docs = tool_outputs.get("retrieved_docs") or []
    if not isinstance(retrieved_docs, list):
        retrieved_docs = [retrieved_docs]
    normalized = [_normalize_doc(doc, idx) for idx, doc in enumerate(retrieved_docs)]
    tool_outputs["retrieved_docs"] = normalized
    scores = [
        score for score in (doc.get("score") for doc in normalized) if score is not None
    ]
    top_score = max(scores) if scores else None
    score_pass = top_score is not None and top_score >= config.SCORE_THRESHOLD
    # 노이즈 문서 주입을 줄이기 위해 score 기반으로만 통과시킨다.
    has_min_docs = len(normalized) >= config.MIN_DOCS
    tool_outputs["rag_relevance_passed"] = bool(score_pass)
    rag_meta = tool_outputs.setdefault("rag_meta", {})
    rag_meta.update(
        {
            "top_score": top_score,
            "has_min_docs": has_min_docs,
            "threshold": config.SCORE_THRESHOLD,
            "normalized": True,
        }
    )
    logger.info(
        "[RAG] RelevanceCheck done: passed=%s top_score=%s min_docs=%s",
        tool_outputs.get("rag_relevance_passed"),
        top_score,
        has_min_docs,
    )

    return state


def relevance_check_decision(state: AgentState) -> str:
    """RelevanceCheck 이후 분기 제어."""

    tool_outputs = state.get("tool_outputs") or {}

    return "inject_docs" if tool_outputs.get("rag_relevance_passed") else "skip_docs"


def retrieved_docs(state: AgentState) -> AgentState:
    """검색 결과 주입."""
    logger.info("[RAG] RetrievedDocs start")
    tool_outputs = state.setdefault("tool_outputs", {})
    retrieved = tool_outputs.get("retrieved_docs") or []
    if not isinstance(retrieved, list):
        retrieved = [retrieved]
    rag_meta = tool_outputs.get("rag_meta") or {}
    if rag_meta.get("normalized") and isinstance(retrieved, list):
        logger.info("[RAG] RetrievedDocs skipped (already normalized)")
        return state
    normalized = [_normalize_doc(doc, idx) for idx, doc in enumerate(retrieved)]
    tool_outputs["retrieved_docs"] = normalized
    logger.info("[RAG] RetrievedDocs done: count=%s", len(normalized))
    return state


builder = StateGraph(AgentState)
builder.add_node("EmbeddingSearch", embedding_search)
builder.add_node("RelevanceCheck", relevance_check)
builder.add_node("RetrievedDocs", retrieved_docs)

builder.set_entry_point("EmbeddingSearch")
builder.add_edge("EmbeddingSearch", "RelevanceCheck")
builder.add_conditional_edges(
    "RelevanceCheck",
    relevance_check_decision,
    {
        "inject_docs": "RetrievedDocs",
        "skip_docs": END,
    },
)
builder.add_edge("RetrievedDocs", END)

graph = builder.compile()
