"""Preprocessing subgraph.

플로우차트의 Preprocessing 영역(CheckType → FileConvert → VisionLLM)
을 담당하는 LangGraph 서브그래프의 구현입니다.
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
    # state 형태의 dict를 넣어 _ensure_fp 호출
    fp = _ensure_fp({"file_processing": fp_like})
    return fp.model_copy(update=update)


def _infer_file_type(input_path_str: Optional[str]) -> str:
    """확장자 기반 판별 로직 보강."""
    if not input_path_str:
        return "text"
    
    # URL 쿼리 스트링 제거 후 순수 확장자 추출
    path_clean = input_path_str.split('?')[0]
    ext = Path(path_clean).suffix.lower()
    
    if ext == ".pdf":
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg"}:
        return "image"
    if ext in {".ppt", ".pptx"}:
        return "ppt"
    return "text"


def _detect_upload_in_messages(messages: List[Any]) -> Optional[str]:
    """
    messages 리스트를 훑어 업로드된 파일의 경로를 반환.
    텍스트와 파일이 섞인 list 형태의 content를 완벽하게 지원합니다.
    """
    if not messages:
        return None
        
    for msg in reversed(messages):
        # 1. 메시지 객체(HumanMessage 등) 혹은 dict에서 content 추출
        content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")

        if not content:
            continue

        # 2. 리스트 형태 처리 (텍스트 + 파일 혼합 입력 시 핵심 로직)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type", "").lower()
                    if t in {"image_url", "image", "file", "file_upload", "attachment"}:
                        # 'file_path' 키를 최우선으로 찾고, 없으면 url/path/name 순으로 탐색
                        path = item.get("file_path") or item.get("url") or item.get("path") or item.get("name")
                        if path: return path
        
        # 3. 단일 dict 형태 처리
        elif isinstance(content, dict):
            t = content.get("type", "").lower()
            if t in {"image_url", "image", "file", "attachment"}:
                path = content.get("file_path") or content.get("url") or content.get("path") or content.get("name")
                if path: return path
        
        # 4. 문자열 형태 처리 (단순 경로 입력 시)
        elif isinstance(content, str):
            if content.startswith(("http", "s3://")) or content.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                return content
                
    return None


def check_type(state: AgentState) -> AgentState:
    print("--- CHECKTYPE START ---")
    messages = state.get("messages", [])
    tool_outputs = state.get("tool_outputs") or {}

    # 1. 파일 경로 찾기 (수동 입력 우선, 그 다음 메시지 탐색)
    detected_path = tool_outputs.get("input_path") or _detect_upload_in_messages(messages)
    
    # 2. 파일 타입 판별
    inferred = _infer_file_type(detected_path)

    # 3. 분류 로직 (텍스트 혼합 여부와 상관없이 파일 성격에 따라 분기)
    if detected_path:
        if inferred in {"pdf", "ppt"}:
            # 변환 과정이 필요한 경우
            category = "mixed_files"
        elif inferred == "image":
            # 바로 비전 분석이 가능한 경우
            category = "image_only"
        else:
            category = "text_only"
    else:
        category = "text_only"

    # 4. State 업데이트
    fp = _ensure_fp(state)
    state["check_result"] = category
    # 발견된 파일 경로(path)를 state에 꼭 저장해줘야 다음 노드에서 사용 가능합니다.
    state["file_processing"] = _model_copy(fp, {
        "file_type": inferred,
        "path": detected_path 
    })
    
    print(f"--- CHECK RESULT: {category} | PATH: {detected_path} ---")
    return state


def file_convert(state: AgentState) -> AgentState:
    print("--- FILECONVERT START ---")
    fp = _ensure_fp(state)
    input_path = fp.path  # CheckType에서 저장한 경로를 사용

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
    print("--- VISIONLLM START ---")
    fp = _ensure_fp(state)
    
    # 1. PDF 변환된 이미지들이 있으면 사용
    images = fp.converted_images
    
    # 2. 만약 PDF 변환이 없었는데(image_only) 단일 이미지 경로가 있다면 리스트로 변환
    if not images and fp.file_type == "image" and fp.path:
        images = [fp.path]

    if not images:
        print("--- SKIP: NO IMAGES TO ANALYZE ---")
        return state

    provider_cfg = state.get("tool_outputs", {}).get("ocr_provider", {"name": "mock"})

    try:
        ocr_result_dict = analyze_images(images, provider_cfg=provider_cfg)
        state["file_processing"] = _model_copy(fp, {"ocr_text": ocr_result_dict})
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