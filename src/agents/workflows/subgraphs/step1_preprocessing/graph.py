"""Preprocessing subgraph.

플로우차트의 Preprocessing 영역(CheckType → FileConvert → VisionLLM)
을 담당하는 LangGraph 서브그래프의 뼈대 구현입니다.
"""

from typing import Literal, List, Dict, Any, Optional, Union
from pathlib import Path
from langgraph.graph import END, StateGraph
from agents.state import AgentState, FileProcessing
from .preprocessing_utils import pdf_to_images
from .vision_llm import analyze_images


def _ensure_fp(state: AgentState) -> FileProcessing:
    """state['file_processing']가 dict/None/인스턴스 여건에 상관없이 안전한 Pydantic 인스턴스 반환."""
    fp_raw = state.get("file_processing")
    if fp_raw is None:
        return FileProcessing(file_type="text")
    if isinstance(fp_raw, FileProcessing):
        return fp_raw
    if isinstance(fp_raw, dict):
        try:
            return FileProcessing.model_validate(fp_raw)
        except Exception:
            return FileProcessing(**fp_raw)
    return FileProcessing(file_type="text")


def _model_copy(
    fp_like: Union[FileProcessing, dict], update: Dict[str, Any]
) -> FileProcessing:
    """dict 또는 FileProcessing 인스턴스 모두 처리하여 model_copy(update=..) 결과 반환."""
    if isinstance(fp_like, FileProcessing):
        fp = fp_like
    elif isinstance(fp_like, dict):
        try:
            fp = FileProcessing.model_validate(fp_like)
        except Exception:
            fp = FileProcessing(**fp_like)
    else:
        fp = FileProcessing(file_type="text")
    return fp.model_copy(update=update)


def _infer_file_type(input_path_str: Optional[str]) -> str:
    """간단한 확장자 기반 판별 로직."""
    if not input_path_str:
        return "text"
    name = input_path_str.split("/")[-1].lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    return "text"


def _detect_upload_in_messages(messages: List[Any]) -> Optional[str]:
    """
    messages 리스트를 훑어 업로드된 파일(이미지/파일)의 경로나 URL을 찾아 반환.
    - 반환값: 발견된 업로드의 경로/URL 또는 None
    """
    if not messages:
        return None
    # 역순으로 최근 메시지부터 검사 (최근 업로드 우선)
    for msg in reversed(messages):
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("type", "").lower()
                        if t in {
                            "image_url",
                            "image",
                            "file",
                            "file_upload",
                            "attachment",
                        }:
                            return (
                                item.get("url") or item.get("path") or item.get("name")
                            )
            # 단일 dict content
            if isinstance(content, dict):
                t = content.get("type", "").lower()
                if t in {"image_url", "image", "file", "attachment"}:
                    return (
                        content.get("url") or content.get("path") or content.get("name")
                    )
        else:
            try:
                content = getattr(msg, "content", None)
            except Exception:
                content = None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("type", "").lower()
                        if t in {
                            "image_url",
                            "image",
                            "file",
                            "file_upload",
                            "attachment",
                        }:
                            return (
                                item.get("url") or item.get("path") or item.get("name")
                            )
            if isinstance(content, dict):
                t = content.get("type", "").lower()
                if t in {"image_url", "image", "file", "attachment"}:
                    return (
                        content.get("url") or content.get("path") or content.get("name")
                    )
            # content가 문자열이고 URL 패턴이면 간단히 URL 반환 (예: pre-signed URL)
            if isinstance(content, str) and (
                content.startswith("http://")
                or content.startswith("https://")
                or content.startswith("s3://")
            ):
                return content
    return None


def check_type(state: AgentState) -> AgentState:
    print("--- CHECKTYPE START ---")
    tool_outputs = state.get("tool_outputs") or {}

    # 1. 수동 입력 경로 확인
    input_path_str = tool_outputs.get("input_path")

    # 2. 채팅 업로드 파일 확인 (input_path가 없을 때만)
    if not input_path_str:
        messages = state.get("messages", [])
        detected_path = _detect_upload_in_messages(messages)
        if detected_path:
            input_path_str = detected_path
            print(f"--- DETECTED FILE: {input_path_str} ---")

    inferred = _infer_file_type(input_path_str)

    if inferred in {"pdf", "ppt"}:
        category = "mixed_files"
    elif inferred == "image":
        category = "image_only"
    else:
        category = "text_only"

    fp = _ensure_fp(state)
    state["check_result"] = category
    state["file_processing"] = _model_copy(fp, {"file_type": inferred})
    return state


def file_convert(state: AgentState) -> AgentState:
    print("--- FILECONVERT (REAL) START ---")
    fp = _ensure_fp(state)

    # 1. CheckType에서 판별된 input_path 가져오기
    input_path = state.get("tool_outputs", {}).get(
        "input_path"
    ) or _detect_upload_in_messages(state.get("messages", []))

    if not input_path:
        print("--- ERROR: NO INPUT PATH FOR CONVERT ---")
        return state


    output_dir = "outputs/temp"
    try:
        real_images = pdf_to_images(input_path, output_dir)
        image_paths = [str(p) for p in real_images]
        state["file_processing"] = _model_copy(fp, {"converted_images": image_paths})
    except Exception as e:
        print(f"--- CONVERT ERROR: {e} ---")

    return state


def vision_llm(state: AgentState) -> AgentState:
    print("--- VISIONLLM (REAL/MOCK) START ---")
    fp = _ensure_fp(state)
    images = fp.converted_images or []

    if not images:
        return state

    # Studio의 입력 가져오기
    provider_cfg = state.get("tool_outputs", {}).get("ocr_provider", {"name": "mock"})

    try:
        ocr_result_dict = analyze_images(images, provider_cfg=provider_cfg)

        state["file_processing"] = _model_copy(
            fp,
            {
                "ocr_text": ocr_result_dict  # {"pages": [...], "full_text": "...", "captions": [...]}
            },
        )
    except Exception as e:
        print(f"--- VISION ERROR: {e} ---")

    return state


def route_by_check_type(
    state: AgentState,
) -> Literal["FileConvert", "VisionLLM", "__end__"]:
    check_result = state.get("check_result", "text_only")
    if check_result == "mixed_files":
        return "FileConvert"
    if check_result == "image_only":
        return "VisionLLM"
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
