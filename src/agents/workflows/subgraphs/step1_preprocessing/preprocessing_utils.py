import os
import logging
import shutil
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POPPLER_PATH = os.getenv("POPPLER_PATH", None)

def get_local_image(ref: str) -> Path:
    """
    입력받은 파일 경로의 존재 여부 확인
    나중에 S3 연동 시 여기서 다운로드 인터페이스를 처리.
    """
    p = Path(ref)
    if not p.exists():
        logger.error(f"파일을 찾을 수 없습니다: {ref}")
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {ref}")
    return p

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200) -> List[Path]:
    """
    PDF의 각 페이지 -> PNG 이미지 변환
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir)
    
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poppler_path = POPPLER_PATH if os.name == "nt" else None

    try:
        logger.info(f"PDF 변환 시작: {pdf_path.name} (DPI={dpi})")
        images = convert_from_path(
            str(pdf_path), 
            dpi=dpi, 
            poppler_path=poppler_path
        )
    except PDFInfoNotInstalledError as e:
        logger.error(f"Poppler 설치 또는 경로를 확인하세요: {e}")
        raise
    except Exception as e:
        logger.exception(f"PDF 변환 중 예상치 못한 오류 발생: {e}")
        raise

    image_paths: List[Path] = []
    for i, image in enumerate(images, start=1):
        # 파일명 형식: 원본이름_page_번호.png
        fname = f"{pdf_path.stem}_page_{i}.png"
        save_path = out_dir / fname
        
        image.convert("RGB").save(save_path, "PNG")
        image_paths.append(save_path)

    logger.info(f"PDF 변환-> {len(image_paths)}개 이미지 생성")
    return image_paths

def preprocess_image(input_path: str, max_dim: int = 2000) -> Path:
    """
    이미지 리사이징 전처리
    """
    in_p = Path(input_path)
    img = Image.open(in_p)
    w, h = img.size

    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(in_p) 
        logger.debug(f"이미지 리사이징 완료: {in_p.name} (scale={scale:.2f})")

    return in_p


if __name__ == "__main__":
    # 1. 프로젝트 루트에 test.pdf를 준비
    TEST_PDF = "test.pdf" 
    OUTPUT_TEMP = "outputs/temp"
    
    try:
        # 파일 존재 확인 및 변환 실행
        pdf_file = get_local_image(TEST_PDF)
        generated_images = pdf_to_images(str(pdf_file), OUTPUT_TEMP)
        
        print("\n" + "="*30)
        print(f"테스트 성공!!!")
        print(f"결과물 위치: {Path(OUTPUT_TEMP).absolute()}")
        print(f"생성된 페이지 수: {len(generated_images)}개")
        print("="*30)
        
    except Exception as e:
        print(f"\n테스트 실패: {e}")