"""Preprocessing subgraph.

플로우차트의 Preprocessing 영역(CheckType → FileConvert → VisionLLM)
을 담당하는 LangGraph 서브그래프의 뼈대 구현입니다.
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from agents.state import AgentState


def check_type(state: AgentState) -> AgentState:
    """파일 형식 체크 (stub)."""
    # TODO: 실제 파일 타입 체크 로직 구현
    # state['check_result'] = "text_only"  # 또는 "image_only", "mixed_files"
    print("---CHECKING INPUT TYPE---")
    state["prev_action"] = "Preprocessing"
    return state


def file_convert(state: AgentState) -> AgentState:
    """PDF/PPT → 이미지 변환 (stub)."""
    print("---CONVERTING FILES---")
    return state


def vision_llm(state: AgentState) -> AgentState:
    """이미지 분석 - OCR + 캡셔닝 (stub)."""
    print("---PROCESSING WITH VISION LLM---")
    state["prev_action"] = "Preprocessing"
    state["next_action"] = "Intent"
    return state


def route_by_check_type(
    state: AgentState,
) -> Literal["FileConvert", "VisionLLM", "__end__"]:
    """파일 형식에 따라 다음 단계를 결정합니다.

    - mixed_files: PDF/PPT 등 변환이 필요한 파일이 포함된 경우 'FileConvert'로 갑니다.
    - image_only: 이미지 파일만 있는 경우, 바로 'VisionLLM'으로 갑니다.
    - text_only: 텍스트만 있는 경우, 전처리 과정 없이 바로 종료합니다.
    """
    check_result = state.get("check_result", "text_only")

    if "mixed_files" in check_result:
        return "FileConvert"
    elif "image_only" in check_result:
        return "VisionLLM"
    else:  # "text_only"
        state["prev_action"] = "Preprocessing"
        state["next_action"] = "Intent"
        return "__end__"


builder = StateGraph(AgentState)
builder.add_node("CheckType", check_type)
builder.add_node("FileConvert", file_convert)
builder.add_node("VisionLLM", vision_llm)

builder.set_entry_point("CheckType")

builder.add_conditional_edges(
    "CheckType",
    route_by_check_type,
    {
        "FileConvert": "FileConvert",
        "VisionLLM": "VisionLLM",
        "__end__": END,
    },
)
builder.add_edge("FileConvert", "VisionLLM")
builder.add_edge("VisionLLM", END)

graph = builder.compile()
