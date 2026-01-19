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
    ocr_text: str = ""
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
                    "ocr_text": f"Mock OCR text for {img_path.name}",
                    "blocks": [
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

    OPENROUTER_API_KEY + OpenAI 호환 Chat API 를 사용한다.
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
            "당신은 수학 문제 분석 전문가입니다. "
            "이미지에서 텍스트와 모든 수식을 추출하여 반드시 아래 JSON 구조로 출력하세요:\n"
            "{\n"
            '  "ocr": [\n'
            "    {\n"
            '      "page": 1,\n'
            '      "ocr_text": "해당 페이지 전체 텍스트",\n'
            '      "blocks": [{"type": "latex", "latex": "수식", "text": "텍스트"}]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "수식은 반드시 'latex' 필드에 LaTeX 문법으로 작성해야 합니다."
        )

        try:
            human = HumanMessage(
                content=[{"type": "text", "text": prompt}] + image_contents
            )
            response = self.model.invoke([human])

            # ChatOpenAI는 일반적으로 단일 문자열 content를 반환한다.
            raw_content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            # 1차 시도: 모델이 잘-구성된 JSON을 반환했다고 가정하고 그대로 파싱
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                # 2차 시도: LaTeX 수식의 '\\' 때문에 JSON 이 깨진 경우를 완화
                clean_content = re.sub(
                    r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
                    r"\\\\",
                    raw_content,
                )
                try:
                    return json.loads(clean_content)
                except json.JSONDecodeError:
                    # 3차 시도: 더 이상 구조화된 JSON 으로 파싱할 수 없으면
                    # 전체 응답을 단일 페이지 OCR 텍스트로 취급하여 반환
                    logger.warning(
                        "Gemini JSON 파싱 실패, raw 텍스트를 단일 페이지로 반환합니다."
                    )
                    return {
                        "ocr": [
                            {
                                "page": 1,
                                "ocr_text": raw_content,
                                "blocks": [],
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
    # 항상 OpenRouter 모델명을 직접 사용한다.
    # 외부에서 어떤 model 값이 들어오더라도, Vision 단계에서는
    # google/gemini-2.5-flash 하나만 고정 사용한다.
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
        resp_raw = provider.analyze(imgs, options={"structured": True})

        raw_ocr_list = resp_raw.get("ocr", [])

        if raw_ocr_list and not isinstance(raw_ocr_list[0], dict):
            raw_ocr_list = [{"page": 1, "ocr_text": str(raw_ocr_list), "blocks": []}]

        pages = []
        for i, p_data in enumerate(raw_ocr_list):
            try:
                p_obj = PageOCR(
                    page=p_data.get("page", i + 1),
                    ocr_text=p_data.get("ocr_text") or p_data.get("text", ""),
                    blocks=[
                        OCRBlock(**b) if isinstance(b, dict) else OCRBlock(text=str(b))
                        for b in p_data.get("blocks", [])
                    ],
                )
                pages.append(p_obj)
            except Exception:
                continue

        formatted_result = {
            "pages": [p.model_dump() for p in pages],
            "full_text": "\n\n".join([p.ocr_text for p in pages if p.ocr_text]),
            "captions": resp_raw.get("image_caption", []),
        }
        return formatted_result

    except Exception as e:
        logger.error(f"비전 분석 프로세스 실패: {e}")
        return {"pages": [], "full_text": f"분석 실패: {str(e)}", "captions": []}


if __name__ == "__main__":
    test_imgs = ["outputs/temp/test_page_1.png"]
    res = analyze_images(test_imgs, {"name": "gemini"})
    print(json.dumps(res, indent=2, ensure_ascii=False))
