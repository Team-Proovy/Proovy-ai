"""
난이도 분류 관련 프롬프트 및 설정.

난이도에 따라 사용하는 LLM 모델:
- easy: Gemini 2.5 Flash (빠르고 저렴)
- medium: Gemini 3 Flash (대부분의 문제는 여기서 해결)
- hard: Gemini 3 Pro (복잡한 문제만, 비용 주의)
"""

from typing import Literal

from agents.prompts.loader import load_prompt, render_prompt
from schema.models import OpenRouterModelName


# 난이도별 모델 매핑
DIFFICULTY_MODEL_MAP = {
    "easy": OpenRouterModelName.GEMINI_25_FLASH,
    "medium": OpenRouterModelName.GEMINI_3_FLASH_PREVIEW,
    "hard": OpenRouterModelName.GEMINI_3_PRO_PREVIEW,
}


DIFFICULTY_CLASSIFIER_SYSTEM_PROMPT = load_prompt(
    "difficulty/classifier_system.txt"
).strip()
DIFFICULTY_CLASSIFIER_USER_PROMPT = load_prompt("difficulty/classifier_user.txt").strip()


def get_model_for_difficulty(
    difficulty: Literal["easy", "medium", "hard"],
) -> OpenRouterModelName:
    """난이도에 맞는 LLM 모델을 반환"""
    return DIFFICULTY_MODEL_MAP.get(
        difficulty,
        OpenRouterModelName.GEMINI_3_FLASH_PREVIEW,
    )


def build_difficulty_classifier_user_prompt(problem_text: str) -> str:
    return render_prompt(
        "difficulty/classifier_user.txt",
        problem_text=problem_text,
    ).strip()


# 난이도별 설명 (UI/로그용)
DIFFICULTY_DESCRIPTIONS = {
    "easy": "기본 문제 - Gemini 2.5 Flash로 빠르게 해결",
    "medium": "중급 문제 - Gemini 3 Flash로 정확하게 해결",
    "hard": "고급 문제 - Gemini 3 Pro로 심층 분석",
}


# 난이도별 예상 응답 시간
DIFFICULTY_EXPECTED_TIME = {
    "easy": "~5초",
    "medium": "~10초",
    "hard": "~20초",
}
