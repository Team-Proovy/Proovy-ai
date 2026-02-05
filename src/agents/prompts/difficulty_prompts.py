"""
난이도 분류 관련 프롬프트 및 설정.

난이도에 따라 사용하는 LLM 모델:
- easy: Gemini 2.5 Flash (빠르고 저렴)
- medium: Gemini 3 Flash (대부분의 문제는 여기서 해결)
- hard: Gemini 3 Pro (복잡한 문제만, 비용 주의)
"""

from typing import Literal
from schema.models import OpenRouterModelName


# 난이도별 모델 매핑
DIFFICULTY_MODEL_MAP = {
    "easy": OpenRouterModelName.GEMINI_25_FLASH,
    "medium": OpenRouterModelName.GEMINI_3_FLASH_PREVIEW,
    "hard": OpenRouterModelName.GEMINI_3_PRO_PREVIEW,
}


def get_model_for_difficulty(difficulty: Literal["easy", "medium", "hard"]) -> OpenRouterModelName:
    """난이도에 맞는 LLM 모델을 반환"""
    return DIFFICULTY_MODEL_MAP.get(difficulty, OpenRouterModelName.GEMINI_3_FLASH_PREVIEW)


# 난이도 분류 시스템 프롬프트
# hard로 잘 안 가도록 기준을 엄격하게 설정
DIFFICULTY_CLASSIFIER_SYSTEM_PROMPT = """You are a difficulty assessor for STEM problems (math, physics, chemistry, programming, etc.).

Your task is to classify the difficulty of the given problem into ONE of these categories:

## EASY (기본 - 대부분의 문제가 여기에 해당)
- Basic arithmetic, simple algebra, direct formula application
- Single-step calculations or straightforward definitions
- Problems that can be solved with one or two simple operations
- Standard textbook examples without complexity
- Examples: "2x + 3 = 7 solve for x", "What is the derivative of x^2?", "Calculate 15% of 200"

## MEDIUM (중급 - 약간 복잡한 문제)
- Multi-step problems requiring 2-4 steps
- Combination of 2-3 concepts
- Word problems with moderate complexity
- Problems requiring setup before solving
- Examples: "A car travels 60km/h for 2 hours then 80km/h for 1.5 hours. Find total distance and average speed."

## HARD (고급 - 매우 복잡한 문제만! 신중하게 선택하세요)
IMPORTANT: Only classify as HARD if the problem meets MULTIPLE of these criteria:
- Requires 5+ distinct solving steps
- Involves proof-based reasoning or mathematical induction
- Combines 4+ advanced concepts from different areas
- Requires creative problem-solving approach not obvious from the problem statement
- Graduate-level or competition mathematics (IMO, Putnam level)
- Involves complex differential equations, advanced linear algebra, or abstract algebra
- Examples: "Prove that there are infinitely many primes", "Solve the system of PDEs with boundary conditions"

## CLASSIFICATION GUIDELINES
1. When in doubt between EASY and MEDIUM, choose EASY
2. When in doubt between MEDIUM and HARD, choose MEDIUM
3. HARD should be rare - less than 5% of problems
4. Most undergraduate-level problems are MEDIUM at most
5. High school level problems are almost always EASY or MEDIUM

Return ONLY one word: EASY, MEDIUM, or HARD"""


# 난이도 분류 사용자 프롬프트 템플릿
DIFFICULTY_CLASSIFIER_USER_PROMPT = """Problem:
{problem_text}

Based on the criteria above, classify this problem's difficulty.
Remember: Choose HARD only for truly complex problems. When uncertain, prefer MEDIUM over HARD.

Difficulty:"""


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
