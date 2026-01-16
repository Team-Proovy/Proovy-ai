"""Preprocessing subgraph.

플로우차트의 Preprocessing 영역(CheckType → FileConvert → VisionLLM)
을 담당하는 LangGraph 서브그래프의 뼈대 구현입니다.
"""

from typing import Literal, List, Dict, Any, Optional, Union
from pathlib import Path
from langgraph.graph import END, StateGraph
from agents.state import AgentState, FileProcessing


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
    # 예상치 못한 타입이면 기본 인스턴스 반환
    return FileProcessing(file_type="text")


def _model_copy(fp_like: Union[FileProcessing, dict], update: Dict[str, Any]) -> FileProcessing:
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
    """간단한 확장자 기반 판별 (pdf/ppt/image/text)."""
    if not input_path_str:
        return "text"
    name = input_path_str.split("/")[-1].lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith((".ppt", ".pptx")):
        return "ppt"
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
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
        # 메시지가 dict로 직렬화된 경우
        if isinstance(msg, dict):
            content = msg.get("content")
            # content가 list 형태로 파일 메타를 포함할 수 있음
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("type", "").lower()
                        if t in {"image_url", "image", "file", "file_upload", "attachment"}:
                            # 우선적으로 file/url 반환
                            return item.get("url") or item.get("path") or item.get("name")
            # 단일 dict content
            if isinstance(content, dict):
                t = content.get("type", "").lower()
                if t in {"image_url", "image", "file", "attachment"}:
                    return content.get("url") or content.get("path") or content.get("name")
        else:
            # BaseMessage-like 객체 (직렬화되지 않은 경우)
            # 접근 가능한 속성: .content 등 (유연하게 처리)
            try:
                content = getattr(msg, "content", None)
            except Exception:
                content = None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("type", "").lower()
                        if t in {"image_url", "image", "file", "file_upload", "attachment"}:
                            return item.get("url") or item.get("path") or item.get("name")
            if isinstance(content, dict):
                t = content.get("type", "").lower()
                if t in {"image_url", "image", "file", "attachment"}:
                    return content.get("url") or content.get("path") or content.get("name")
            # content가 문자열이고 URL 패턴이면 간단히 URL 반환 (예: pre-signed URL)
            if isinstance(content, str) and (content.startswith("http://") or content.startswith("https://") or content.startswith("s3://")):
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
    """PDF/PPT -> 이미지 변환 (테스트/모크용 간단 구현)."""
    print("--- DEBUG: SAFE FILECONVERT (MOCK) ---")
    fp = _ensure_fp(state)
    # 실제 변환 로직은 preprocessing_utils.ppt_to_pdf / pdf_to_images 등으로 대체
    mock_images = ["mock_page_1.png", "mock_page_2.png"]
    state["file_processing"] = _model_copy(fp, {"converted_images": mock_images})
    return state


def vision_llm(state: AgentState) -> AgentState:
    """
    VisionLLM 통합 자리:
    - 현재는 mock 결과를 채워 테스트가 가능하도록 구현
    - 실제 통합 시: 여기에 Vision-capable LLM 호출 코드를 넣으면 됩니다.
      예: vision_model.invoke([HumanMessage(content=[{"type":"image_url","url":...}])])
    """
    print("--- DEBUG: SAFE VISIONLLM (MOCK) ---")
    fp = _ensure_fp(state)
    images = fp.converted_images or []

    # 실제: Vision LLM 호출 예시 
    #     from your_project.models import vision_model
    #     # Prepare prompt to extract text & LaTeX
    #     prompt = "이미지에서 모든 텍스트와 수식을 LaTeX 형식으로 추출해 JSON으로 반환해줘."
    #     for img in images:
    #         res = vision_model.invoke([HumanMessage(content=[{"type":"image_url","url": img}]), HumanMessage(content=prompt)])
    #         # parse res and append to pages/full_text
 

    # 현재는 mock OCR 결과로 채움
    mock_pages = [{"image": img, "ocr": f"샘플 텍스트 ({img})", "latex": ""} for img in images]
    mock_full_text = " ".join(p["ocr"] for p in mock_pages) or "VisionLLM 통합 테스트용 가짜 텍스트입니다."

    state["file_processing"] = _model_copy(fp, {"ocr_text": {"pages": mock_pages, "full_text": mock_full_text}})
    return state


def route_by_check_type(state: AgentState) -> Literal["FileConvert", "VisionLLM", "__end__"]:
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
