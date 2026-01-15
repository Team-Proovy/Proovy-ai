"""Preprocessing subgraph.

플로우차트의 Preprocessing 영역(CheckType → FileConvert → VisionLLM)
을 담당하는 LangGraph 서브그래프의 뼈대 구현입니다.
"""

from langgraph.graph import StateGraph

from agents.state import AgentState


def check_type(state: AgentState) -> AgentState:
    """파일 형식 체크 (stub)."""

    return state


def file_convert(state: AgentState) -> AgentState:
    """PDF/PPT → 이미지 변환 (stub)."""

    return state


def vision_llm(state: AgentState) -> AgentState:
    """이미지 분석 - OCR + 캡셔닝 (stub)."""

    return state


builder = StateGraph(AgentState)
builder.add_node("CheckType", check_type)
builder.add_node("FileConvert", file_convert)
builder.add_node("VisionLLM", vision_llm)

builder.set_entry_point("CheckType")
builder.add_edge("CheckType", "FileConvert")
builder.add_edge("FileConvert", "VisionLLM")

graph = builder.compile()
