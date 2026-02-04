from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, List, Optional, Tuple

from agents.state import AgentState
from agents.workflows.utils import (
    call_model,
    extract_ocr_text,
    recent_user_context,
    safe_json_loads,
)
from schema.models import OpenRouterModelName


PROBLEM_MARKER_PATTERN = re.compile(r"(?m)^\s*(?:문제\s*)?\d{1,3}[\.\)]\s+")
OPTION_LIKE_PATTERN = re.compile(
    r"^\s*(?:[A-D]|[가-라]|[A-D가-라]형|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|\d+\s*/\s*\d+|\d+)\s*$"
)
LEADING_NUMBER_PATTERN = re.compile(
    r"^\s*(?:문제\s*)?(\d{1,3}|[①-⑳])\s*(?:번|[.\)])?"
)
PROBLEM_BODY_HINT_PATTERN = re.compile(r"(보기|\[[0-9]+\s*점\]|[=<>]|[①-⑩]|Σ|∑|√)")
CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
EXPLANATION_LEADING_NUMBER_PATTERN = re.compile(
    r"^\s*(?:문제\s*)?(\d{1,3}|[①-⑳])\s*[.\)]?\s*"
)
SKIP_KEYWORDS = ("참고", "해설", "정답", "예제", "풀이", "유의사항")
CIRCLED_NUMBER_MAP = {ch: idx + 1 for idx, ch in enumerate(CIRCLED_NUMBERS)}
CIRCLED_NUMBER_MAP.update({str(i): i for i in range(1, 101)})
ANSWER_PATTERNS = [
    re.compile(r"정답\s*[:：]\s*([^\n]+)"),
    re.compile(r"정답은\s*([^\n]+)"),
    re.compile(r"\(정답\s*[:：]?\s*([^)]+)\)"),
]
SUMMARY_PATTERNS = [
    re.compile(r"^\s*summary\s*[:：]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*요약\s*[:：]?\s*$"),
]
LATEX_COMMAND_PATTERN = re.compile(r"\\[a-zA-Z]+")
HANGUL_PATTERN = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
OPTION_MARKER_PATTERN = re.compile(r"^\s*([①-⑳]|[ㄱ-ㅎ]|[A-D]|\d{1,3})[.)]?\s*")
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*(?:page|p\.?)\s*\d+(?:\s*/\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*\d+\s*(?:쪽|p)\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–—]*\s*\d+\s*[-–—]*\s*$"),
]
MATH_TOKEN_PATTERN = re.compile(r"[=<>±*/^_∑Σ√∫∞πθλΔ∆]")
HEADER_CANDIDATE_MAX_LEN = 60
LATEX_COMPARE_REPLACEMENTS = {
    r"\times": "×",
    r"\cdot": "·",
    r"\pi": "π",
    r"\infty": "∞",
    r"\sum": "∑",
    r"\ln": "ln",
    r"\to": "→",
    r"\le": "≤",
    r"\ge": "≥",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\sim": "~",
}
SUMMARY_PATTERNS = [
    re.compile(r"^\s*summary\s*[:：]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*요약\s*[:：]?\s*$"),
]


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        text = block.get("text")
        latex = block.get("latex")
    else:
        text = getattr(block, "text", None)
        latex = getattr(block, "latex", None)
    parts: List[str] = []
    if text:
        parts.append(str(text).strip())
    if latex:
        latex_value = str(latex).strip()
        if latex_value and latex_value not in parts:
            parts.append(latex_value)
    return "\n".join([p for p in parts if p]).strip()


def _normalize_header_line(line: str) -> str:
    normalized = re.sub(r"\s+", " ", line.strip())
    return normalized.lower()


def _is_page_number_line(line: str) -> bool:
    if not line:
        return False
    return any(pattern.match(line) for pattern in PAGE_NUMBER_PATTERNS)


def _is_header_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) < 2 or len(stripped) > HEADER_CANDIDATE_MAX_LEN:
        return False
    if PROBLEM_MARKER_PATTERN.match(stripped):
        return False
    if OPTION_LIKE_PATTERN.match(stripped):
        return False
    if _is_page_number_line(stripped):
        return False
    if "\\" in stripped:
        return False
    if MATH_TOKEN_PATTERN.search(stripped):
        return False
    return True


def _collect_repeated_headers(page_blocks: List[List[str]]) -> set[str]:
    if len(page_blocks) < 2:
        return set()
    per_page: List[set[str]] = []
    for blocks in page_blocks:
        candidates: set[str] = set()
        for block in blocks:
            for line in str(block).splitlines():
                if _is_header_candidate(line):
                    candidates.add(_normalize_header_line(line))
        per_page.append(candidates)
    counts: Counter[str] = Counter()
    for candidates in per_page:
        counts.update(candidates)
    total_pages = len(page_blocks)
    threshold = 2 if total_pages < 4 else max(2, (total_pages + 1) // 2)
    return {line for line, count in counts.items() if count >= threshold}


def _strip_headers_from_text(text: str, repeated_headers: set[str]) -> str:
    lines = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_page_number_line(stripped):
            continue
        if repeated_headers and _normalize_header_line(stripped) in repeated_headers:
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def _extract_option_marker(line: str) -> Optional[str]:
    match = OPTION_MARKER_PATTERN.match(line)
    if not match:
        return None
    return match.group(1)


def _is_latex_source_line(line: str) -> bool:
    if not line:
        return False
    return bool(LATEX_COMMAND_PATTERN.search(line))


def _normalize_math_text_for_compare(text: str) -> str:
    normalized = str(text)
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        r"(\1)/(\2)",
        normalized,
    )
    for key, value in LATEX_COMPARE_REPLACEMENTS.items():
        normalized = normalized.replace(key, value)
    normalized = re.sub(r"[\s{}]", "", normalized)
    return normalized


def _page_blocks_from_state(state: AgentState) -> List[List[str]]:
    file_processing = state.get("file_processing")
    ocr_blocks = None
    if isinstance(file_processing, dict):
        ocr_blocks = file_processing.get("ocr_blocks")
    elif file_processing is not None:
        ocr_blocks = getattr(file_processing, "ocr_blocks", None)

    pages = None
    if isinstance(ocr_blocks, dict):
        pages = ocr_blocks.get("pages")
    elif isinstance(ocr_blocks, list):
        pages = ocr_blocks
    elif ocr_blocks is not None:
        pages = getattr(ocr_blocks, "pages", None)

    if not isinstance(pages, list):
        return []

    page_blocks: List[List[str]] = []
    for page in pages:
        if isinstance(page, dict):
            blocks = page.get("blocks") or []
        else:
            blocks = getattr(page, "blocks", None) or []
        if not isinstance(blocks, list):
            blocks = [blocks]
        block_texts: List[str] = []
        for block in blocks:
            text = _block_text(block)
            if text:
                block_texts.append(text)
        if block_texts:
            page_blocks.append(block_texts)
    repeated_headers = _collect_repeated_headers(page_blocks)
    cleaned_pages: List[List[str]] = []
    for blocks in page_blocks:
        cleaned_blocks: List[str] = []
        for block in blocks:
            cleaned = _strip_headers_from_text(block, repeated_headers)
            if cleaned:
                cleaned_blocks.append(cleaned)
        if cleaned_blocks:
            cleaned_pages.append(cleaned_blocks)
    return cleaned_pages


def _is_probable_problem_start(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if os.getenv("SOLUTION_USE_KEYWORD_FILTER") == "1":
        if any(keyword in stripped for keyword in SKIP_KEYWORDS):
            return False
    if not PROBLEM_MARKER_PATTERN.match(stripped):
        return False
    if "문제" in stripped or "보기" in stripped:
        return True
    after = PROBLEM_MARKER_PATTERN.sub("", stripped, count=1).strip()
    if not after:
        return False
    if len(after) <= 3:
        return False
    if OPTION_LIKE_PATTERN.match(after):
        return False
    return True


def _looks_like_problem_body(text: str) -> bool:
    if not text:
        return False
    if PROBLEM_BODY_HINT_PATTERN.search(text):
        return True
    if re.search(r"\d", text) and len(text) >= 40:
        return True
    return False


def _classify_blocks_with_llm(blocks: List[str]) -> Optional[List[int]]:
    if not blocks:
        return None
    items = [
        {"id": idx, "text": text[:800]}
        for idx, text in enumerate(blocks)
        if text and text.strip()
    ]
    if not items:
        return None
    system_prompt = (
        "You classify OCR blocks into problem vs non-problem.\n"
        "problem: question statement, 보기, 선택지, 조건/수식.\n"
        "non-problem: 제목/소제목, 유의사항, 머리말/꼬리말, 출제/채점/응시 안내.\n"
        "Return ONLY JSON: {\"problem_ids\":[1,2,...]}."
    )
    user_prompt = "blocks:\n" + json.dumps(items, ensure_ascii=False)
    raw = call_model(OpenRouterModelName.GPT_5_MINI, system_prompt, user_prompt)
    payload = safe_json_loads(raw) if isinstance(raw, str) else {}
    ids = payload.get("problem_ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list):
        return None
    result: List[int] = []
    for value in ids:
        if isinstance(value, int):
            result.append(value)
        elif isinstance(value, str) and value.isdigit():
            result.append(int(value))
    return sorted({idx for idx in result if 0 <= idx < len(blocks)})


def _collect_problems(state: AgentState) -> List[str]:
    page_blocks = _page_blocks_from_state(state)
    problems_with_meta: List[Tuple[int, int, str]] = []
    for page_idx, blocks in enumerate(page_blocks):
        if not blocks:
            continue
        has_marker = any(_is_probable_problem_start(text) for text in blocks)
        if not has_marker:
            page_text = "\n".join(blocks).strip()
            use_llm_filter = os.getenv("SOLUTION_USE_LLM_FILTER", "1") == "1"
            if use_llm_filter and page_text:
                problem_ids = _classify_blocks_with_llm(blocks)
                if problem_ids is not None:
                    if not problem_ids:
                        continue
                    selected = [blocks[idx] for idx in problem_ids if blocks[idx].strip()]
                    if selected:
                        problems_with_meta.append((page_idx, 0, "\n".join(selected).strip()))
                        continue
            if page_text and _looks_like_problem_body(page_text):
                problems_with_meta.append((page_idx, 0, page_text))
            continue

        current_parts: List[str] = []
        current_meta: Optional[Tuple[int, int]] = None
        for block_idx, text in enumerate(blocks):
            stripped = text.strip()
            if not stripped:
                continue
            if _is_probable_problem_start(stripped):
                if current_parts:
                    problems_with_meta.append(
                        (current_meta[0], current_meta[1], "\n".join(current_parts).strip())
                    )
                current_parts = [stripped]
                current_meta = (page_idx, block_idx)
            else:
                if current_parts:
                    current_parts.append(stripped)
        if current_parts:
            problems_with_meta.append(
                (current_meta[0], current_meta[1], "\n".join(current_parts).strip())
            )

    if problems_with_meta:
        if os.getenv("SOLUTION_SORT_BY_NUMBER", "1") == "1":
            extracted: List[Tuple[int, int, int, Optional[int], str]] = []
            for idx, (page_idx, block_idx, text) in enumerate(problems_with_meta):
                num = _extract_problem_number(text)
                extracted.append((page_idx, block_idx, idx, num, text))
            nums = [item[3] for item in extracted if item[3] is not None]
            if nums and len(nums) >= len(extracted) // 2:
                extracted.sort(
                    key=lambda x: (
                        x[3] if x[3] is not None else 10**6,
                        x[0],
                        x[1],
                        x[2],
                    )
                )
                return [item[4] for item in extracted]
        return [item[2] for item in problems_with_meta]

    fallback = extract_ocr_text(state) or recent_user_context(state)
    if fallback:
        return [fallback.strip()]
    return []


def _extract_problem_number(text: str) -> Optional[int]:
    match = LEADING_NUMBER_PATTERN.match(text.strip())
    if not match:
        return None
    raw = match.group(1)
    if raw.isdigit():
        return int(raw)
    return CIRCLED_NUMBER_MAP.get(raw)


def _extract_explanation_number(text: str) -> Optional[int]:
    match = EXPLANATION_LEADING_NUMBER_PATTERN.match(text.strip())
    if not match:
        return None
    raw = match.group(1)
    if raw.isdigit():
        return int(raw)
    return CIRCLED_NUMBER_MAP.get(raw)


def _strip_leading_number(text: str) -> str:
    return EXPLANATION_LEADING_NUMBER_PATTERN.sub("", text, count=1).strip()


def _strip_problem_marker(text: str) -> str:
    stripped = text.strip()
    cleaned = PROBLEM_MARKER_PATTERN.sub("", stripped, count=1).strip()
    return cleaned or stripped


def _extract_answer_from_explanation(explanation: str) -> str:
    if not explanation:
        return ""
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(explanation)
        if match:
            answer = match.group(1).strip()
            return answer.rstrip(".")
    return ""


def _clean_problem_text(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines()]
    option_markers: set[str] = set()
    for line in lines:
        if not line:
            continue
        if _is_latex_source_line(line):
            continue
        marker = _extract_option_marker(line)
        if marker:
            option_markers.add(marker)
    cleaned: List[str] = []
    seen_normalized: set[str] = set()
    for line in lines:
        if not line:
            continue
        if any(pattern.match(line) for pattern in SUMMARY_PATTERNS):
            continue
        if any(keyword in line for keyword in SKIP_KEYWORDS):
            if not PROBLEM_BODY_HINT_PATTERN.search(line) and not re.search(r"\d", line):
                continue
        if _is_latex_source_line(line):
            marker = _extract_option_marker(line)
            if marker and marker in option_markers:
                continue
            if "\\text{" in line:
                continue
            if HANGUL_PATTERN.search(line):
                continue
        normalized = _normalize_math_text_for_compare(line)
        if normalized and normalized in seen_normalized:
            continue
        if normalized:
            seen_normalized.add(normalized)
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _align_explanations_to_problems(
    problems: List[str],
    explanations: List[str],
) -> List[str]:
    problem_map: dict[int, int] = {}
    for idx, problem in enumerate(problems):
        num = _extract_problem_number(problem)
        if num is not None:
            problem_map[num] = idx

    parsed: List[Tuple[Optional[int], str]] = []
    for explanation in explanations:
        num = _extract_explanation_number(explanation)
        parsed.append((num, explanation))

    numbered = [item for item in parsed if item[0] is not None]
    if numbered and problem_map:
        out = ["해설을 생성하지 못했습니다."] * len(problems)
        for num, explanation in parsed:
            if num is None:
                continue
            idx = problem_map.get(num)
            if idx is None:
                continue
            cleaned = _strip_leading_number(explanation)
            out[idx] = cleaned or explanation

        fallback = [
            _strip_leading_number(text) or text
            for num, text in parsed
            if num is None
        ]
        fill_idx = 0
        for idx in range(len(out)):
            if out[idx] != "해설을 생성하지 못했습니다.":
                continue
            if fill_idx >= len(fallback):
                break
            out[idx] = fallback[fill_idx]
            fill_idx += 1
        return out

    cleaned = [ex.strip() for ex in explanations]
    if len(cleaned) < len(problems):
        cleaned.extend(["해설을 생성하지 못했습니다."] * (len(problems) - len(cleaned)))
    return cleaned[: len(problems)]


def _build_pdf_entries(
    problems: List[str],
    explanations: List[str],
) -> List[dict]:
    entries: List[dict] = []
    for problem, explanation in zip(problems, explanations):
        number = _extract_problem_number(problem)
        cleaned_problem = _clean_problem_text(problem)
        display_problem = (
            _strip_problem_marker(cleaned_problem)
            if number is not None
            else cleaned_problem.strip()
        )
        answer = _extract_answer_from_explanation(explanation)
        entries.append(
            {
                "number": number,
                "problem": display_problem,
                "answer": answer,
                "explanation": explanation,
            }
        )
    return entries

