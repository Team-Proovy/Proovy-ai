"""다중 문제 분리 모듈

OCR 텍스트나 사용자 입력에서 개별 문제들을 식별하고 분리합니다.
"""

import json
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.prompts.splitter_prompts import SPLIT_SYSTEM_PROMPT, build_split_user_prompt
from core.llm import get_model
from schema.models import OpenRouterModelName


class ProblemSplit(BaseModel):
    """문제 분리 결과"""

    is_multiple: bool = Field(description="다중 문제 여부")
    problems: List[str] = Field(default_factory=list, description="분리된 개별 문제들")
    problem_numbers: List[str] = Field(
        default_factory=list, description="문제 번호 목록 (예: ['1번', '2번'])"
    )


def _quick_check_multiple_problems(text: str) -> bool:
    """빠른 휴리스틱 검사: 다중 문제 가능성이 있는지 확인"""
    if not text or len(text.strip()) < 20:
        return False

    # 패턴 매칭으로 빠르게 확인
    patterns = [
        r"(?:^|\n)\s*[1-9]\d*\s*[.)]\s*",  # 1. 또는 1)
        r"(?:^|\n)\s*문제\s*[1-9]",  # 문제 1
        r"(?:^|\n)\s*[1-9]\d*번",  # 1번
        r"(?:^|\n)\s*Q[1-9]",  # Q1
        r"(?:^|\n)\s*Problem\s*[1-9]",  # Problem 1
    ]

    match_count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if len(matches) >= 2:
            return True
        match_count += len(matches)

    return match_count >= 2


def _rule_based_split(text: str) -> Optional[ProblemSplit]:
    """규칙 기반 문제 분리 (LLM 호출 전 시도)"""

    # 패턴: "1.", "2." 또는 "1)", "2)"
    pattern1 = r"(?:^|\n)\s*([1-9]\d*)\s*[.)]\s*"
    # 패턴: "문제 1", "문제 2"
    pattern2 = r"(?:^|\n)\s*문제\s*([1-9]\d*)[.:]?\s*"
    # 패턴: "1번", "2번"
    pattern3 = r"(?:^|\n)\s*([1-9]\d*)번[.:]?\s*"

    for pattern in [pattern1, pattern2, pattern3]:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if len(matches) >= 2:
            problems = []
            problem_numbers = []

            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                problem_text = text[start:end].strip()

                if problem_text:
                    problems.append(problem_text)
                    num = match.group(1)
                    problem_numbers.append(f"{num}번")

            if len(problems) >= 2:
                return ProblemSplit(
                    is_multiple=True,
                    problems=problems,
                    problem_numbers=problem_numbers,
                )

    return None


def split_problems(text: str, use_llm: bool = True) -> ProblemSplit:
    """
    텍스트에서 개별 문제들을 분리합니다.

    Args:
        text: OCR 텍스트 또는 사용자 입력
        use_llm: LLM 사용 여부 (False면 규칙 기반만 사용)

    Returns:
        ProblemSplit: 분리된 문제 목록
    """
    # 빈 텍스트 처리
    if not text or len(text.strip()) < 10:
        return ProblemSplit(is_multiple=False, problems=[text or ""], problem_numbers=[""])

    # 빠른 휴리스틱 검사
    if not _quick_check_multiple_problems(text):
        return ProblemSplit(is_multiple=False, problems=[text], problem_numbers=[""])

    # 규칙 기반 분리 시도
    rule_result = _rule_based_split(text)
    if rule_result:
        print(f"---SPLITTER: Rule-based split found {len(rule_result.problems)} problems---")
        return rule_result

    # LLM 기반 분리
    if not use_llm:
        return ProblemSplit(is_multiple=False, problems=[text], problem_numbers=[""])

    print("---SPLITTER: Using LLM for problem splitting---")

    try:
        model = get_model(OpenRouterModelName.GPT_5_MINI).with_config(tags=["skip_stream"])

        messages = [
            SystemMessage(content=SPLIT_SYSTEM_PROMPT),
            HumanMessage(content=build_split_user_prompt(text[:3000])),
        ]

        response = model.invoke(messages)
        content = response.content

        # JSON 파싱
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
        result = ProblemSplit(**data)

        if result.is_multiple and len(result.problems) >= 2:
            print(f"---SPLITTER: LLM split found {len(result.problems)} problems---")
            return result

    except Exception as e:
        print(f"---SPLITTER: LLM error: {e}---")

    # 기본값 반환
    return ProblemSplit(is_multiple=False, problems=[text], problem_numbers=[""])


def get_current_problem_text(
    problems: List[str], current_index: int, problem_numbers: List[str]
) -> str:
    """현재 문제 텍스트와 번호를 포맷팅하여 반환"""
    if not problems or current_index >= len(problems):
        return ""

    problem_text = problems[current_index]
    number = (
        problem_numbers[current_index]
        if current_index < len(problem_numbers)
        else f"{current_index + 1}번"
    )

    # 이미 번호가 포함되어 있으면 그대로 반환
    if number in problem_text[:20]:
        return problem_text

    return f"[{number} 문제]\n{problem_text}"
