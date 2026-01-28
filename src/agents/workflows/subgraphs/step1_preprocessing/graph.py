"""Preprocessing subgraph.

플로우차트의 Preprocessing 영역(CheckType → FileConvert → VisionLLM)
을 담당하는 LangGraph 서브그래프의 뼈대 구현입니다.
"""

from typing import Literal, List, Dict, Any, Optional
from pathlib import Path

from langgraph.graph import END, StateGraph
from agents.state import AgentState, FileProcessing

from agents.workflows.subgraphs.step1_preprocessing import preprocessing_utils
from agents.workflows.subgraphs.step1_preprocessing.vision_llm import analyze_images

# --- 헬퍼들  ---

def _model_copy(fp: FileProcessing, update: Dict[str, Any]) -> FileProcessing:
    """Pydantic v2 환경: model_copy(update=...)로 새 인스턴스 반환."""
    return fp.model_copy(update=update)


def _ensure_fp(state: AgentState) -> FileProcessing:
    fp = state.get("file_processing")
    if fp is None:
        fp = FileProcessing(file_type="text")
    return fp


def _is_s3_uri(uri: Optional[str]) -> bool:
    return isinstance(uri, str) and uri.startswith("s3://")


def _infer_file_type(input_path_str: Optional[str]) -> str:
    """
    입력 문자열(로컬 경로 또는 s3 URI) 또는 None을 받아
    'pdf' | 'image' | 'text' 반환.
    """
    if not input_path_str:
        return "text"
    cleaned = input_path_str.split("?")[0].split("#")[0]
    name = Path(cleaned.split("/")[-1]).name  # s3 또는 local 모두 처리
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    return "text"


def _localize_input_path(input_path_str: str, tmp_dir: Path) -> Path:
    """
    s3:// URI면 preprocessing_utils.download_s3_to_local을 호출해 로컬 Path 반환.
    로컬이면 Path로 바로 반환.
    """
    if _is_s3_uri(input_path_str):
        if not hasattr(preprocessing_utils, "download_s3_to_local"):
            raise RuntimeError(
                "S3 입력을 처리하려면 'preprocessing_utils.download_s3_to_local(s3_uri, target_dir)' "
                "함수가 구현되어 있어야 합니다."
            )
        return preprocessing_utils.download_s3_to_local(input_path_str, tmp_dir)
    return Path(input_path_str)


def _extract_path_from_file_item(item: Any) -> Optional[str]:
    if not item:
        return None
    if isinstance(item, (str, Path)):
        return str(item)
    if isinstance(item, dict):
        for key in ("path", "file_path", "url", "s3_uri", "filename", "name", "file_name"):
            value = item.get(key)
            if value:
                return str(value)
    return None


def _collect_input_paths(state: AgentState, tool_outputs: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    input_path_str = tool_outputs.get("input_path")
    if input_path_str:
        candidates.append(str(input_path_str))

    input_files = state.get("input_files") or tool_outputs.get("input_files") or []
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    if isinstance(input_files, list):
        for item in input_files:
            extracted = _extract_path_from_file_item(item)
            if extracted:
                candidates.append(extracted)

    # 중복 제거(순서 유지)
    seen = set()
    unique: List[str] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _primary_input_path(state: AgentState, tool_outputs: Dict[str, Any]) -> Optional[str]:
    paths = _collect_input_paths(state, tool_outputs)
    return paths[0] if paths else None


def _infer_primary_type_from_paths(paths: List[str]) -> str:
    if any(_infer_file_type(p) == "pdf" for p in paths):
        return "pdf"
    if any(_infer_file_type(p) == "image" for p in paths):
        return "image"
    return "text"


def _pick_preferred_path(paths: List[str], preferred: set[str]) -> Optional[str]:
    for path in paths:
        if _infer_file_type(path) in preferred:
            return path
    return paths[0] if paths else None



def check_type(state: AgentState) -> AgentState:
    """파일 유형 판별 및 file_processing.file_type 업데이트."""
    print("--- CHECKTYPE ---")
    tool_outputs = state.get("tool_outputs") or {}
    paths = _collect_input_paths(state, tool_outputs)
    inferred = _infer_primary_type_from_paths(paths) if paths else _infer_file_type(None)
    if inferred == "pdf":
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
    """
    PDF -> 페이지별 이미지 변환.
    이 노드는 'mixed_files'일 때 호출.
    """
    print("--- FILECONVERT ---")
    tool_outputs = state.get("tool_outputs") or {}
    paths = _collect_input_paths(state, tool_outputs)
    input_path_str = _pick_preferred_path(paths, {"pdf"})
    if not input_path_str:
        return state

    tmp_dir_base = Path(tool_outputs.get("tmp_dir", "/tmp/lang_preprocess"))
    tmp_dir_name = Path(input_path_str.split("/")[-1]).stem
    tmp_dir = tmp_dir_base / tmp_dir_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_input = _localize_input_path(input_path_str, tmp_dir)

    fp = _ensure_fp(state)
    suffix = local_input.suffix.lower()

    if suffix != ".pdf":
        return state
    pdf_path = local_input

    # PDF -> 페이지 이미지
    image_paths: List[Path] = preprocessing_utils.pdf_to_images(pdf_path, tmp_dir)
    converted_images = [str(p) for p in image_paths]

    state["file_processing"] = _model_copy(fp, {"converted_images": converted_images})
    return state


def vision_llm(state: AgentState) -> AgentState:
    """
    이미지 전처리 → Vision LLM 기반 구조화 OCR → ocr_blocks 저장.
    """
    print("--- VISIONLLM (OCR + CAPTION) ---")
    tool_outputs = state.get("tool_outputs") or {}
    fp = _ensure_fp(state)
    images = fp.converted_images or []

    if not images:
        paths = _collect_input_paths(state, tool_outputs)
        images = [p for p in paths if _infer_file_type(p) == "image"]

    if not images:
        return state

    tmp_dir_base = Path(tool_outputs.get("tmp_dir", "/tmp/lang_preprocess"))
    prepared_images: List[str] = []
    for img_ref in images:
        local_img = _localize_input_path(img_ref, tmp_dir_base)
        try:
            proc_img = preprocessing_utils.preprocess_image(
                local_img, local_img.with_suffix(".proc.png")
            )
            prepared_images.append(str(proc_img))
        except Exception:
            prepared_images.append(str(local_img))

    if not prepared_images:
        return state

    provider_cfg = tool_outputs.get("ocr_provider") or {"name": "gemini"}
    if not isinstance(provider_cfg, dict):
        provider_cfg = {"name": str(provider_cfg)}
    provider_name = str(provider_cfg.get("name", "")).lower()
    if provider_name in {"mathpix", "tesseract"}:
        provider_cfg = {**provider_cfg, "name": "gemini"}

    try:
        ocr_result_dict = analyze_images(prepared_images, provider_cfg=provider_cfg)
        state["file_processing"] = _model_copy(fp, {"ocr_blocks": ocr_result_dict})
    except Exception as e:
        print(f"--- VISION ERROR: {e} ---")
    return state


def route_by_check_type(state: AgentState) -> Literal["FileConvert", "VisionLLM", "__end__"]:
    """파일 형식에 따라 다음 단계를 결정합니다.

    - mixed_files: PDF/PPT 등 변환이 필요한 파일이 포함된 경우 'FileConvert'로 갑니다.
    - image_only: 이미지 파일만 있는 경우, 바로 'VisionLLM'으로 갑니다.
    - text_only: 텍스트만 있는 경우, 전처리 과정 없이 바로 종료합니다.
    """
    check_result = state.get("check_result", "text_only")
    if check_result == "mixed_files":
        return "FileConvert"
    elif check_result == "image_only":
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
