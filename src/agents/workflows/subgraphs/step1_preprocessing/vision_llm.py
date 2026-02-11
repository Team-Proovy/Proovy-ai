import json
import logging
import mimetypes
import re
from base64 import b64encode
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from core.settings import settings
from agents.prompts.vision_ocr_prompt import VISION_OCR_PROMPT

logger = logging.getLogger(__name__)


def _load_prompt_text() -> str:
    return VISION_OCR_PROMPT


class OCRBlock(BaseModel):
    type: str = Field(
        "text",
        description="header, text, math_inline, math_display, equation, latex, page_num 등",
    )
    text: str = Field("", description="추출된 텍스트 또는 수식 내용")
    bbox: Optional[List[int]] = Field(None, description="[ymin, xmin, ymax, xmax]")
    latex: Optional[str] = Field(None, description="수식 블록일 경우 LaTeX 코드")


class PageOCR(BaseModel):
    page: int = 1
    blocks: List[OCRBlock] = Field(default_factory=list)


class StructuredOCRResponse(BaseModel):
    ocr: List[PageOCR] = Field(default_factory=list)
    image_caption: List[Dict[str, Any]] = Field(default_factory=list)


class VisionProvider:
    def analyze(self, images: List[Path], options: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class MockVisionProvider(VisionProvider):
    def analyze(self, images: List[Path], options: Dict[str, Any]) -> Dict[str, Any]:
        ocr_list = []
        for idx, img_path in enumerate(images, start=1):
            ocr_list.append(
                {
                    "page": idx,
                    "blocks": [
                        {
                            "type": "text",
                            "text": f"Mock OCR text for {img_path.name}",
                        },
                        {
                            "type": "latex",
                            "text": "a/b",
                            "latex": "\\frac{a}{b}",
                            "bbox": [0, 0, 0, 0],
                        }
                    ],
                }
            )
        return {"ocr": ocr_list, "image_caption": []}


class OpenRouterGeminiVisionProvider(VisionProvider):
    """OpenRouter의 google/gemini-2.5-flash 모델을 사용하는 Vision Provider.
    """

    def __init__(self, model_name: str = "google/gemini-2.5-flash"):
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다.")

        # OpenRouter의 OpenAI 호환 엔드포인트를 사용하는 ChatOpenAI 인스턴스 생성
        self.model = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

    def analyze(self, images: List[Path], options: Dict[str, Any]) -> Dict[str, Any]:
        image_contents: List[Dict[str, Any]] = []
        for img_path in images:
            if not img_path.exists():
                continue
            mime_type, _ = mimetypes.guess_type(img_path)
            mime = mime_type or "image/png"
            b64 = b64encode(img_path.read_bytes()).decode("utf-8")
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        prompt = _load_prompt_text()

        try:
            human = HumanMessage(
                content=[{"type": "text", "text": prompt}] + image_contents
            )
            response = self.model.invoke([human])

           
            raw_content = response.content
            if isinstance(raw_content, list):
                parts: List[str] = []
                for chunk in raw_content:
                    if isinstance(chunk, dict):
                        parts.append(
                            str(
                                chunk.get("text")
                                or chunk.get("content")
                                or chunk.get("data")
                                or ""
                            )
                        )
                    else:
                        parts.append(str(chunk))
                raw_content = "".join(parts)
            elif not isinstance(raw_content, str):
                raw_content = str(raw_content)
            def _candidate_strings(text: str) -> List[str]:
                stripped = text.strip()
                candidates: List[str] = []
                if stripped:
                    candidates.append(stripped)
                if "```" in text:
                    for part in text.split("```"):
                        part = part.strip()
                        if not part or part.lower().startswith("json"):
                            continue
                        candidates.append(part)

                def _extract_json_blocks(src: str) -> List[str]:
                    blocks: List[str] = []
                    depth = 0
                    start_idx: Optional[int] = None
                    in_str = False
                    str_char: Optional[str] = None
                    escape = False
                    for idx, ch in enumerate(src):
                        if in_str:
                            if escape:
                                escape = False
                                continue
                            if ch == "\\":
                                escape = True
                                continue
                            if ch == str_char:
                                in_str = False
                                str_char = None
                            continue
                        if ch in {"'", '"'}:
                            in_str = True
                            str_char = ch
                            continue
                        if ch == "{":
                            if depth == 0:
                                start_idx = idx
                            depth += 1
                            continue
                        if ch == "}" and depth > 0:
                            depth -= 1
                            if depth == 0 and start_idx is not None:
                                blocks.append(src[start_idx : idx + 1].strip())
                                start_idx = None
                    return blocks

                candidates.extend(_extract_json_blocks(text))

                # 중복 제거 (순서 유지)
                seen = set()
                uniq: List[str] = []
                for item in candidates:
                    if item in seen:
                        continue
                    seen.add(item)
                    uniq.append(item)

                def _candidate_score(candidate: str) -> tuple[int, int, int]:
                    has_ocr_key = bool(re.search(r"""["']ocr["']\s*:""", candidate))
                    has_ocr_word = bool(re.search(r"""["']ocr["']""", candidate))
                    return (1 if has_ocr_key else 0, 1 if has_ocr_word else 0, len(candidate))

                # "ocr" 키 포함 후보를 우선 처리하고, 길이가 긴 후보를 선호
                scored = [(idx, _candidate_score(item), item) for idx, item in enumerate(uniq)]
                scored.sort(key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True)
                return [item for _, _, item in scored]

            def _try_load_json(text: str) -> Dict[str, Any] | None:
                # 1차: 원문 그대로 파싱
                for candidate in _candidate_strings(text):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

                def _escape_latex_in_json(src: str) -> str:
                    def _repl(match: re.Match) -> str:
                        quote = match.group(1)
                        inner = match.group(2)
                        inner_escaped = re.sub(r'(?<!\\)\\', r"\\\\", inner)
                        return f'"latex": {quote}{inner_escaped}{quote}'

                    return re.sub(r'"latex"\s*:\s*("|\')([\s\S]*?)\1', _repl, src)

                # 2차: LaTeX 필드만 타겟팅하여 이스케이프 처리
                clean_text = _escape_latex_in_json(text)
                for candidate in _candidate_strings(clean_text):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

                # 3차: "ocr" 키를 포함하는 후보 강제 시도
                ocr_match = re.search(r'(\{[\s\S]*?["\']ocr["\'][\s\S]*?\})', text)
                if ocr_match:
                    try:
                        return json.loads(ocr_match.group(1))
                    except json.JSONDecodeError:
                        pass

                return None

            parsed = _try_load_json(raw_content)
            if parsed is not None:
                return parsed

            # 3차 시도: 더 이상 구조화된 JSON 으로 파싱할 수 없으면
            # 전체 응답을 단일 페이지 OCR 텍스트로 취급하여 반환
            logger.warning(
                "Gemini JSON 파싱 실패, raw 텍스트를 단일 페이지로 반환합니다."
            )
            return {
                "ocr": [
                    {
                        "page": 1,
                        "blocks": [{"type": "text", "text": raw_content}],
                    }
                ],
                "image_caption": [],
            }
        except Exception:
            preview = raw_content[:200] if "raw_content" in locals() else ""
            logger.error(f"OpenRouter Gemini 응답 원본 확인: {preview}...")
            logger.exception("OpenRouter Gemini 분석 중 오류 발생")
            raise


def get_provider(cfg: Dict[str, Any]) -> VisionProvider:
    name = str(cfg.get("name", "mock")).lower()

    model_name = "google/gemini-2.5-flash"

    print(
        f"--- DEBUG: name={name}, model={model_name}, openrouter_key_exists={bool(settings.OPENROUTER_API_KEY)} ---"
    )

    if name == "gemini" and settings.OPENROUTER_API_KEY:
        return OpenRouterGeminiVisionProvider(model_name=model_name)

    return MockVisionProvider()


def analyze_images(
    image_paths: List[str], provider_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    provider_cfg = provider_cfg or {"name": "mock"}
    provider = get_provider(provider_cfg)
    imgs = [Path(p) for p in image_paths if p]

    try:
        def _normalize_ocr_list(raw_ocr: Any) -> List[Any]:
            if raw_ocr is None:
                return []
            if isinstance(raw_ocr, list):
                return raw_ocr
            return [raw_ocr]

        def _parse_pages(raw: Any) -> tuple[List[PageOCR], List[Dict[str, Any]]]:
            if not isinstance(raw, dict):
                return [], []
            raw_ocr_list = _normalize_ocr_list(raw.get("ocr"))
            if raw_ocr_list and not isinstance(raw_ocr_list[0], dict):
                raw_ocr_list = [
                    {
                        "page": idx + 1,
                        "blocks": [{"type": "text", "text": str(item)}],
                    }
                    for idx, item in enumerate(raw_ocr_list)
                ]

            parsed_pages: List[PageOCR] = []
            extra_captions: List[Dict[str, Any]] = []
            for i, p_data in enumerate(raw_ocr_list):
                try:
                    if isinstance(p_data, dict):
                        if isinstance(p_data.get("image_caption"), list):
                            extra_captions.extend(p_data.get("image_caption") or [])
                        has_blocks = bool(p_data.get("blocks"))
                        has_text = bool(p_data.get("text") or p_data.get("ocr_text"))
                        if p_data.get("caption") and not has_blocks and not has_text:
                            extra_captions.append(
                                {
                                    "page": p_data.get("page", i + 1),
                                    "caption": p_data.get("caption"),
                                }
                            )
                            continue
                        page_num = p_data.get("page", i + 1)
                        blocks_data = p_data.get("blocks") or []
                        if not isinstance(blocks_data, list):
                            blocks_data = [blocks_data]
                        if not blocks_data:
                            fallback_text = (
                                p_data.get("text") or p_data.get("ocr_text") or ""
                            )
                            if fallback_text:
                                blocks_data = [
                                    {"type": "text", "text": str(fallback_text)}
                                ]
                    else:
                        page_num = i + 1
                        blocks_data = [{"type": "text", "text": str(p_data)}]

                    # 블록 정규화 및 중복 제거
                    normalized_blocks: List[Dict[str, Any]] = []
                    seen_keys: set[tuple] = set()
                    for block in blocks_data:
                        if not isinstance(block, dict):
                            block = {"type": "text", "text": str(block)}
                        block_type = str(block.get("type") or "text").strip().lower()
                        text = str(block.get("text") or "")
                        latex = str(block.get("latex") or "")
                        bbox = block.get("bbox") if isinstance(block.get("bbox"), list) else None
                        if latex and not text:
                            block_type = "latex"
                        elif latex and block_type in {"", "text"}:
                            # Choice prefix like "ㄱ." or "①" should be treated as math block
                            if not re.search(r"[가-힣A-Za-z]", text):
                                block_type = "math_inline"
                        # If this is a math block but latex is missing, reuse text as latex
                        if block_type in {"latex", "equation", "math_inline", "math_display"} and not latex and text:
                            latex = text
                        normalized = {
                            "type": block_type,
                            "text": text,
                            "latex": latex,
                            "bbox": bbox,
                        }
                        key = (
                            normalized.get("type") or "",
                            (normalized.get("text") or "").strip(),
                            (normalized.get("latex") or "").strip(),
                            tuple(normalized.get("bbox") or []),
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        normalized_blocks.append(normalized)
                    blocks_data = normalized_blocks

                    p_obj = PageOCR(
                        page=page_num,
                        blocks=[
                            OCRBlock(**b)
                            if isinstance(b, dict)
                            else OCRBlock(text=str(b))
                            for b in blocks_data
                        ],
                    )
                    parsed_pages.append(p_obj)
                except Exception:
                    continue
            return parsed_pages, extra_captions

        def _pick_caption_text(raw: Any, page_num: int) -> str:
            if not isinstance(raw, dict):
                return ""
            captions = raw.get("image_caption")
            if not isinstance(captions, list):
                return ""
            for item in captions:
                if (
                    isinstance(item, dict)
                    and item.get("page") == page_num
                    and item.get("caption")
                ):
                    return str(item.get("caption")).strip()
            for item in captions:
                if isinstance(item, dict) and item.get("caption"):
                    return str(item.get("caption")).strip()
            return ""

        resp_raw = provider.analyze(imgs, options={"structured": True})
        captions = resp_raw.get("image_caption", []) if isinstance(resp_raw, dict) else []
        pages, extra_captions = _parse_pages(resp_raw)
        if extra_captions:
            captions = list(captions or []) + extra_captions

        if not pages and imgs:
            logger.warning("OCR 결과가 비어 있어 페이지 단위로 재시도합니다.")
            fallback_pages: List[PageOCR] = []
            fallback_captions: List[Dict[str, Any]] = []
            for idx, img in enumerate(imgs, start=1):
                try:
                    single_raw = provider.analyze([img], options={"structured": True})
                except Exception as exc:
                    logger.warning(f"페이지 OCR 재시도 실패(page={idx}): {exc}")
                    continue

                if isinstance(single_raw, dict):
                    fallback_captions.extend(single_raw.get("image_caption") or [])

                single_pages, extra_single = _parse_pages(single_raw)
                if extra_single:
                    fallback_captions.extend(extra_single)
                if not single_pages:
                    caption_text = _pick_caption_text(single_raw, idx)
                    if caption_text:
                        single_pages = [
                            PageOCR(
                                page=idx,
                                blocks=[OCRBlock(type="text", text=caption_text)],
                            )
                        ]

                for page in single_pages:
                    if not page.page:
                        page.page = idx
                    fallback_pages.append(page)

            if fallback_pages:
                pages = fallback_pages
            if not captions and fallback_captions:
                captions = fallback_captions

        formatted_result = {
            "pages": [p.model_dump() for p in pages],
            "captions": captions,
        }
        return formatted_result

    except Exception as e:
        logger.error(f"비전 분석 프로세스 실패: {e}")
        return {"pages": [], "captions": [], "error": f"분석 실패: {str(e)}"}


if __name__ == "__main__":
    test_imgs = ["outputs/temp/test_page_1.png"]
    res = analyze_images(test_imgs, {"name": "gemini"})
    print(json.dumps(res, indent=2, ensure_ascii=False))
