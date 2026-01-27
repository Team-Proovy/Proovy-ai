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

logger = logging.getLogger(__name__)


class OCRBlock(BaseModel):
    type: str = Field("text", description="header, text, equation, latex, page_num 등")
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

        prompt = (
            "당신은 다양한 종류의 이미지를 읽고 이해하는 멀티모달 OCR & 캡셔닝 전문가입니다. "
            "주어진 이미지를 보고 텍스트, 수식, 레이아웃 정보를 구조화된 JSON 한 개로 반환하세요.\n\n"
            "반드시 아래 조건을 지키세요:\n"
            "1. 오직 하나의 JSON 객체만 출력합니다. 앞뒤에 설명 문장은 쓰지 않습니다.\n"
            '2. 최상위에는 반드시 "ocr"(리스트), "image_caption"(리스트) 키를 포함합니다.\n'
            "3. 각 페이지는 blocks로만 구성하며, blocks에 페이지의 모든 텍스트/수식을 포함합니다.\n"
            "4. 입력 이미지 개수와 동일한 길이의 ocr 리스트를 반환하고, 순서를 유지합니다.\n"
            "5. 텍스트가 거의 없더라도 blocks는 비우지 말고 빈 문자열 블록을 하나 넣습니다.\n\n"
            "6. 블록은 문단/문제 단위로 최대한 묶어서 반환합니다.\n\n"
            "출력 JSON 스키마 예시는 다음과 같습니다:\n"
            "{\n"
            '  "ocr": [\n'
            "    {\n"
            '      "page": 1,\n'
            '      "blocks": [\n'
            "        {\n"
            '          "type": "header | text | latex | equation | page_num | figure | table",\n'
            '          "text": "블록 내 텍스트 또는 수식 설명",\n'
            '          "latex": "수식이 있는 경우 LaTeX 표현, 없으면 빈 문자열",\n'
            '          "bbox": [ymin, xmin, ymax, xmax]\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            '  "image_caption": [\n'
            "    {\n"
            '      "page": 1,\n'
            '      "caption": "이 페이지 또는 전체 이미지에 대한 자연어 설명"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Instructions in English:\n"
            "- Always return a single JSON object with keys `ocr` and `image_caption`.\n"
            "- For each page, return only `page` and `blocks` (do not include `ocr_text`).\n"
            "- All readable text must appear in `blocks`.\n"
            "- The `ocr` array length must match the number of input images (keep order).\n"
            "- If text is missing, include one block with empty text instead of an empty list.\n"
            "- `image_caption` must be written in Korean.\n"
            "- Prefer paragraph or question-level blocks; do not split one sentence into many blocks.\n"
            "- Split the page into `blocks` with `type`, `text`, optional `latex`, "
            "and `bbox` = [ymin, xmin, ymax, xmax] in pixels.\n"
            "- If there is no math, set `latex` to an empty string.\n"
            "- If you are unsure, still follow the schema and use empty strings or empty arrays instead of omitting keys.\n"
        )

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
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidates.append(text[start : end + 1].strip())
                # 중복 제거 (순서 유지)
                seen = set()
                uniq: List[str] = []
                for item in candidates:
                    if item in seen:
                        continue
                    seen.add(item)
                    uniq.append(item)
                return uniq

            def _try_load_json(text: str) -> Dict[str, Any] | None:
                # 1차: 원문 그대로 파싱
                for candidate in _candidate_strings(text):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                # 2차: LaTeX 수식의 '\\' 때문에 JSON 이 깨진 경우를 완화
                clean_text = re.sub(
                    r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
                    r"\\\\",
                    text,
                )
                for candidate in _candidate_strings(clean_text):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
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
