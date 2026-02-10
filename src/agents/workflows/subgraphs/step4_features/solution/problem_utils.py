from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, List, Optional, Tuple

from agents.state import AgentState
from .pdf_utils import (
    latex_to_unicode_shared as _render_latex_to_plain
)

# 정규식 패턴 강화
PROBLEM_MARKER_PATTERN = re.compile(r"(?m)^\s*(?:문제\s*)?(\d{1,3})[\.\)]")
CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_NUMBER_MAP = {ch: idx + 1 for idx, ch in enumerate(CIRCLED_NUMBERS)}
CIRCLED_NUMBER_MAP.update({str(i): i for i in range(1, 101)})
PLACEHOLDER_RE = re.compile(r"[□■¤]")
CHOICE_ONLY_RE = re.compile(r"^\s*([①-⑳]|[ㄱ-ㅎ]|[0-9]+[.)])\s*$")
CHOICE_LINE_RE = re.compile(r"^\s*([①-⑳]|[ㄱ-ㅎ])[\.)]?\s*")
VIEW_LINE_RE = re.compile(r"^\s*<?보기>?\s*$")
SIN_FRACTION_RE = re.compile(r"sin\s*x\s*2\s*\+\s*e\^\{?x\}?", re.IGNORECASE)
LN_FRACTION_RE = re.compile(r"ln\s*\(?\s*2n\^3\s*\)?\s*1\s*\+\s*n\^2", re.IGNORECASE)

def _fix_common_missing_fractions(text: str) -> str:
    if not text:
        return text
    # Avoid double-wrapping if LaTeX fraction already present
    if "\\frac" in text:
        return text
    updated = text
    if SIN_FRACTION_RE.search(updated):
        updated = SIN_FRACTION_RE.sub(r"$\\frac{\\sin x}{2+e^x}$", updated)
    if LN_FRACTION_RE.search(updated):
        updated = LN_FRACTION_RE.sub(r"$\\frac{\\ln(2n^3)}{1+n^2}$", updated)
    return updated

def _strip_choice_lines(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if VIEW_LINE_RE.match(stripped):
            continue
        if CHOICE_LINE_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept).strip()

def _global_clean(text: str) -> str:
    """OCR 노이즈 및 비정상 기호 정제"""
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[¤□■¶§©®™†‡\xa0]', ' ', text)
    return re.sub(r" +", " ", text).strip()

def _normalize_for_compare(text: str) -> str:
    """[구조 개선] 텍스트와 LaTeX 모두를 위한 통합 비교 키 생성 함수"""
    if not text: return ""
    # 1. 원문 정규화 및 원문 숫자 변환
    t = unicodedata.normalize("NFKC", text)
    for circle, num in CIRCLED_NUMBER_MAP.items():
        if not str(circle).isdigit():
            t = t.replace(circle, str(num))

    # 2. LaTeX 구조 간소화 (중괄호/공백/기본 변형 흡수)
    t = re.sub(r"\\(?:d|t)?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", t)
    t = re.sub(r"\\sqrt(?:\[[^\]]+\])?\{([^}]+)\}", r"sqrt\1", t)
    t = t.replace("{", "").replace("}", "")
    t = t.replace("^", "").replace("_", "")

    # 3. LaTeX 명령어 및 서식 명령 제거
    t = re.sub(r"\\[a-zA-Z]+", "", t)

    # 4. 순수 의미(숫자, 한글, 영문)만 남김
    t = re.sub(r"[^0-9a-zA-Z가-힣]", "", t)
    return t.lower()


def _latex_inline_for_match(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"\\(?:d|t)?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", t)
    t = re.sub(r"\\sqrt(?:\[[^\]]+\])?\{([^}]+)\}", r"sqrt\1", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = t.replace("{", "").replace("}", "")
    t = re.sub(r"\s+", "", t)
    return t


def _wrap_latex(text: str) -> str:
    if not text:
        return ""
    if text.startswith("$"):
        return text
    if text.startswith(r"\(") and text.endswith(r"\)"):
        return text
    if text.startswith(r"\[") and text.endswith(r"\]"):
        return text
    return f"${text}$"


def _extract_choice_prefix(text: str) -> str:
    if not text:
        return ""
    match = re.match(r"^\s*([①-⑳]|[ㄱ-ㅎ]|[0-9]+[.)])\s*", text)
    if match:
        return match.group(0).strip() + " "
    return ""

def _extract_problem_number(text: str) -> Optional[int]:
    clean_t = _global_clean(text)
    match = PROBLEM_MARKER_PATTERN.search(clean_t)
    if not match: match = re.search(r"([①-⑳])", clean_t)
    if not match: return None
    val = match.group(1)
    return int(val) if val.isdigit() else CIRCLED_NUMBER_MAP.get(val)

def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        t, l = block.get("text") or "", block.get("latex") or ""
        block_type = (block.get("type") or "").strip().lower()
    else:
        t, l = getattr(block, "text", "") or "", getattr(block, "latex", "") or ""
        block_type = (getattr(block, "type", "") or "").strip().lower()
    
    t = _global_clean(t)
    if t and not l:
        t = _fix_common_missing_fractions(t)
    if l:
        # [구조 개선] 원문 보존을 위해 LaTeX 원본 유지 ($ 델리미터 보존)
        l_wrapped = _wrap_latex(l)
        if not t:
            return l_wrapped

        # 수식 전용 블록은 라벨만 보존하고 LaTeX로 표시
        if block_type in {"equation", "latex", "math_inline", "math_display"}:
            prefix = _extract_choice_prefix(t)
            rest = t[len(prefix):].strip() if prefix else t
            if not re.search(r"[가-힣]", rest):
                return f"{prefix}{l_wrapped}" if prefix else l_wrapped

        # 텍스트가 이미 LaTeX를 포함하는 경우 중복 추가 방지
        if l_wrapped in t or l in t:
            return t

        # 텍스트에 수식이 이미 표현되어 있으면 중복 추가를 피함
        t_inline = re.sub(r"\s+", "", t)
        l_inline = _latex_inline_for_match(l)
        if l_inline and l_inline in t_inline:
            return t

        # 설명 텍스트가 있는 경우는 보존하고, 필요 시 LaTeX만 추가
        has_explain_text = bool(re.search(r"[가-힣A-Za-z]", t))
        t_norm = _normalize_for_compare(t)
        l_norm = _normalize_for_compare(l)
        if t_norm and l_norm and (l_norm in t_norm or t_norm in l_norm):
            return t if has_explain_text else l_wrapped

        return f"{t} {l_wrapped}"
    return t

def _collect_problems(state: AgentState) -> List[str]:
    # [Fix] Major Null check for file_processing
    file_processing = state.get("file_processing")
    if not file_processing: return []
    
    ocr_blocks = file_processing.get("ocr_blocks") if isinstance(file_processing, dict) else getattr(file_processing, "ocr_blocks", None)
    if not ocr_blocks: return []
    
    pages = ocr_blocks.get("pages", []) if isinstance(ocr_blocks, dict) else getattr(ocr_blocks, "pages", [])
    if not pages: return []
    
    problems_map = {}
    # [Fix] Critical: Move problem_seen_norms outside the page loop to avoid KeyError
    problem_seen_norms = {} 
    last_num = None
    
    def _block_type(block: Any) -> str:
        if isinstance(block, dict):
            return (block.get("type") or "").strip().lower()
        return (getattr(block, "type", "") or "").strip().lower()

    def _raw_text(block: Any) -> str:
        if isinstance(block, dict):
            return str(block.get("text") or "")
        return str(getattr(block, "text", "") or "")

    def _raw_latex(block: Any) -> str:
        if isinstance(block, dict):
            return str(block.get("latex") or "")
        return str(getattr(block, "latex", "") or "")

    def _is_math_block(block: Any) -> bool:
        if _raw_latex(block):
            return True
        return _block_type(block) in {"latex", "equation", "math_inline", "math_display"}

    def _is_math_text(text: str) -> bool:
        if not text:
            return False
        return ("$" in text) or ("\\" in text)

    for page in pages:
        blocks = page.get("blocks", []) if isinstance(page, dict) else getattr(page, "blocks", [])
        i = 0
        while i < len(blocks):
            block = blocks[i]
            raw_text = _raw_text(block)
            raw_latex = _raw_latex(block)
            text = ""

            # Placeholder merge: replace □ with next math blocks' LaTeX
            if raw_text and PLACEHOLDER_RE.search(raw_text) and not raw_latex:
                merged_text = raw_text
                j = i + 1
                while PLACEHOLDER_RE.search(merged_text) and j < len(blocks) and _is_math_block(blocks[j]):
                    latex = _raw_latex(blocks[j])
                    if latex:
                        merged_text = PLACEHOLDER_RE.sub(_wrap_latex(latex), merged_text, count=1)
                    j += 1
                text = _global_clean(merged_text)
                i = j
            else:
                # Merge choice-only line with following math block (e.g., "ㄱ." + latex)
                if raw_text and CHOICE_ONLY_RE.match(raw_text) and (i + 1) < len(blocks) and _is_math_block(blocks[i + 1]):
                    next_block = blocks[i + 1]
                    next_latex = _raw_latex(next_block) or _raw_text(next_block)
                    if next_latex:
                        merged_text = f"{raw_text.strip()} {_wrap_latex(next_latex)}"
                        text = _global_clean(merged_text)
                        i += 2
                    else:
                        text = _block_text(block)
                        i += 1
                else:
                    text = _block_text(block)
                    i += 1

            if not text:
                continue

            num = _extract_problem_number(text)
            target_num = num if num else last_num

            if target_num is not None:
                if target_num not in problems_map:
                    problems_map[target_num] = []
                    problem_seen_norms[target_num] = set()

                norm = _normalize_for_compare(text)
                # 부분 일치로도 중복 판단 (유사 선지 제거 강화)
                if norm:
                    if _is_math_text(text) or len(norm) < 4:
                        if norm in problem_seen_norms[target_num]:
                            continue
                    else:
                        if any(norm in seen or seen in norm for seen in problem_seen_norms[target_num]):
                            continue

                problems_map[target_num].append(text)
                problem_seen_norms[target_num].add(norm)
                if num: last_num = num
            else:
                if 0 not in problems_map:
                    problems_map[0] = []
                    problem_seen_norms[0] = set()

                norm = _normalize_for_compare(text)
                if norm:
                    if _is_math_text(text) or len(norm) < 4:
                        if norm in problem_seen_norms[0]:
                            continue
                    else:
                        if any(norm in seen or seen in norm for seen in problem_seen_norms[0]):
                            continue
                problems_map[0].append(text)
                if norm:
                    problem_seen_norms[0].add(norm)
                
    if not problems_map: return []
    
    include_header = os.getenv("SOLUTION_INCLUDE_HEADER", "1") == "1"
    if 0 in problems_map:
        header_text_list = problems_map.pop(0)
        if include_header:
            sorted_keys = sorted(problems_map.keys())
            if sorted_keys:
                first_key = sorted_keys[0]
                problems_map[first_key] = header_text_list + problems_map[first_key]
            else:
                problems_map[0] = header_text_list

    sorted_keys = sorted(problems_map.keys())
    return ["\n".join(problems_map[k]).strip() for k in sorted_keys]

def _align_explanations_to_problems(problems, explanations):
    """[구조 개선] 정렬 단계에서는 단순 매핑만 수행 (표현 변환 개입 금지)"""
    return [str(e).strip() for e in explanations[:len(problems)]]

def _get_problem_title(text: str) -> str:
    """문제 본문에서 첫 줄을 추출하여 요약 제목 생성 (표현용이므로 즉시 변환)"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: return "문제"
    title = re.sub(r'^\d+[\.\)]\s*', '', lines[0])
    title = _render_latex_to_plain(title) # 출력용 제목으로 즉시 변환
    return title[:60] + ("..." if len(title) > 60 else "")

def _build_pdf_entries(problems, explanations):
    """
    [구조 개선] PDF 엔트리 생성.
    NOTE: 여기서는 데이터 정합성 유지를 위해 LaTeX 원문을 보존합니다.
    실제 유니코드/평문 수식으로의 표현 변환은 graph.py의 최종 출력 단계에서 단 한번 수행됩니다.
    """
    entries = []
    ANS_RE = re.compile(r"(?:정답|답)[\s:：]*([①-⑳A-E0-9\(\)/ㄱㄴㄷㄹㅁ\+\-\.\*]+)")
    
    for p, e in zip(problems, explanations):
        num = _extract_problem_number(p)
        e_clean = _global_clean(e)
        
        ans_match = ANS_RE.search(e_clean)
        ans = ans_match.group(1).strip() if ans_match else "-"
        
        expl = ANS_RE.sub("", e_clean).strip()
        expl = re.sub(r"(?m)^\d+[\.\)]\s*", "", expl)
        expl = re.sub(r"^해설\s*[:：]\s*", "", expl)
        expl = _fix_common_missing_fractions(expl)
        
        original_text = p.strip()
        include_choices = os.getenv("SOLUTION_INCLUDE_CHOICES", "1").strip().lower() not in {"0", "false", "no", "off"}
        if not include_choices:
            original_text = _strip_choice_lines(original_text)
        entries.append({
            "number": num,
            "title": _get_problem_title(p),
            "original": original_text,      # 원본 LaTeX 보존 (선택지 제거 옵션 적용)
            "answer": ans,                  # 원본 보존 (graph.py에서 변환)
            "explanation": expl.strip()     # 원본 보존 (graph.py에서 변환)
        })
    return entries
