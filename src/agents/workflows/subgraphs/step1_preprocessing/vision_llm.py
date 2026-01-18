import os
import json
import logging
import mimetypes 
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def check_genai():
    try:
        import google.generativeai as genai
        return True, genai
    except (ImportError, ModuleNotFoundError):
        return False, None

HAS_GENAI, genai = check_genai()

from pydantic import BaseModel, Field

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
            ocr_list.append({
                "page": idx,
                "ocr_text": f"Mock OCR text for {img_path.name}",
                "blocks": [{"type": "latex", "text": "a/b", "latex": "\\frac{a}{b}", "bbox": [0,0,0,0]}]
            })
        return {"ocr": ocr_list, "image_caption": []}

class GeminiVisionProvider(VisionProvider):
    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        success, lib = check_genai()
        if not success:
            raise RuntimeError("Gemini 라이브러리가 없습니다.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
        lib.configure(api_key=api_key)
        self.model = lib.GenerativeModel(model_name)

    def analyze(self, images: List[Path], options: Dict[str, Any]) -> Dict[str, Any]:
        image_parts = []
        for img_path in images:
            if not img_path.exists(): continue
            mime_type, _ = mimetypes.guess_type(img_path)
            image_parts.append({
                "mime_type": mime_type or "image/png",
                "data": img_path.read_bytes()
            })

        prompt = """당신은 수학 문제 분석 전문가입니다. 
        이미지에서 텍스트와 모든 수식을 추출하여 반드시 아래 JSON 구조로 출력하세요:
        {
          "ocr": [
            {
              "page": 1,
              "ocr_text": "해당 페이지 전체 텍스트",
              "blocks": [{"type": "latex", "latex": "수식", "text": "텍스트"}]
            }
          ]
        }
        수식은 반드시 'latex' 필드에 LaTeX 문법으로 작성해야 합니다."""

        try:
            response = self.model.generate_content(
                [prompt] + image_parts,
                generation_config={"response_mime_type": "application/json", "temperature": 0.1}
            )
            
            # LaTeX 수식의 '\'가 JSON의 escape rule과 충돌하여 발생하는 JSONDecodeError 방지
            # 유효하지 않은 escape sequence(예: \sum, \int)만 골라 '\\'로 치환함
            raw_content = response.text
            clean_content = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', raw_content)
            
            return json.loads(clean_content)
        except Exception as e:
            logger.error(f"Gemini 응답 원본 확인: {response.text[:200]}...")
            logger.exception("Gemini 분석 중 오류 발생")
            raise

def get_provider(cfg: Dict[str, Any]) -> VisionProvider:
    name = str(cfg.get("name", "mock")).lower()
    model_name = cfg.get("model", "models/gemini-2.5-flash")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    current_has_genai, _ = check_genai()
    print(f"--- DEBUG: name={name}, api_key_exists={bool(api_key)}, has_genai={current_has_genai} ---")
    
    if name == "gemini":
        if current_has_genai and api_key:
            return GeminiVisionProvider(model_name=model_name)
    
    return MockVisionProvider()


def analyze_images(image_paths: List[str], provider_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                    blocks=[OCRBlock(**b) if isinstance(b, dict) else OCRBlock(text=str(b)) 
                            for b in p_data.get("blocks", [])]
                )
                pages.append(p_obj)
            except Exception:
                continue

        formatted_result = {
            "pages": [p.model_dump() for p in pages],
            "full_text": "\n\n".join([p.ocr_text for p in pages if p.ocr_text]),
            "captions": resp_raw.get("image_caption", [])
        }
        return formatted_result

    except Exception as e:
        logger.error(f"비전 분석 프로세스 실패: {e}")
        return {"pages": [], "full_text": f"분석 실패: {str(e)}", "captions": []}

if __name__ == "__main__":
    test_imgs = ["outputs/temp/test_page_1.png"]
    res = analyze_images(test_imgs, {"name": "gemini"})
    print(json.dumps(res, indent=2, ensure_ascii=False))