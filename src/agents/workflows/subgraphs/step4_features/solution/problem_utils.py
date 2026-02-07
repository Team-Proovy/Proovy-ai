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

# 정규식 패턴 강화: 
# 1. 키워드가 있는 경우 (예: 문제 1, Q1) -> 기호는 옵션
# 2. 키워드가 없는 경우 -> 기호(. , ) , ])는 필수이며 앞뒤로 공백이나 줄 시작/끝이 있어야 함
# (?<!\S)는 앞에 공백이 아닌 문자가 오지 않음을 의미(줄 시작 또는 공백)하여 가변 길이 후방 탐색 오류를 방지합니다.
PROBLEM_MARKER_PATTERN = re.compile(
    r"(?i)(?:(?:문제|Q|No|Task|Step|\[|#)\s*\d{1,3}[\.\)\]]?|(?<!\S)\d{1,3}[\.\)\]](?!\S))"
)

CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_NUMBER_MAP = {ch: idx + 1 for idx, ch in enumerate(CIRCLED_NUMBERS)}
CIRCLED_NUMBER_MAP.update({str(i): i for i in range(1, 101)})

def _global_clean(text: str) -> str:
    """OCR 노이즈 및 비정상 기호 정제"""
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[¤□■¶§©®™†‡\xa0]', ' ', text)
    return re.sub(r" +", " ", text).strip()

def _normalize_for_compare(text: str) -> str:
    """[구조 개선] 텍스트와 LaTeX 모두를 위한 통합 비교 키 생성 함수"""
    if not text: return ""
    t = unicodedata.normalize("NFKC", text)
    for circle, num in CIRCLED_NUMBER_MAP.items():
        if not str(circle).isdigit():
            t = t.replace(circle, str(num))
    
    t = re.sub(r"\\[a-zA-Z]+", "", t) 
    
    t = re.sub(r"[^0-9a-zA-Z가-힣]", "", t)
    return t.lower()

def extract_problem_number(text: str) -> Optional[int]:
    """텍스트에서 문제 번호를 추출."""
    if not text: return None
    clean_t = _global_clean(text)
    
    match = PROBLEM_MARKER_PATTERN.search(clean_t)
    if match:
        num_match = re.search(r"\d+", match.group(0))
        if num_match:
            return int(num_match.group(0))
            
    circle_match = re.search(r"([①-⑳])", clean_t)
    if circle_match:
        return CIRCLED_NUMBER_MAP.get(circle_match.group(1))
        
    return None

def split_text_into_problems(text: str) -> List[str]:
    """텍스트 내의 문제 마커를 기준으로 텍스트를 분할합니다."""
    if not text:
        return []

    pattern = re.compile(
        r"((?:(?:문제|Q|No|Task|Step|\[|#)\s*\d{1,3}[\.\)\]]?|(?<!\S)\d{1,3}[\.\)\]](?!\S)))", re.IGNORECASE
    )
    parts = pattern.split(text)

    problems = []
    # 첫 번째 파트 처리: 본문이 어느 정도 있는 경우만 추가
    first_part = parts[0].strip()
    if first_part and len(first_part) > 2:
        problems.append(first_part)

    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        combined = (marker + content).strip()
        
        # [핵심 수정] 유령 문제 방지: 
        # 마커(예: '문제 2.')만 있고 뒤에 본문이 거의 없으면 유효한 문제로 보지 않음
        if combined and len(combined) > len(marker.strip()) + 2:
            problems.append(combined)

    if not problems and text.strip():
        return [text.strip()]

    # 최종 필터링: 공백 제거 후 최소 5자 이상인 유효한 문제만 반환
    return [p for p in problems if p and len(p.strip()) > 5]


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        t, l = block.get("text") or "", block.get("latex") or ""
    else:
        t, l = getattr(block, "text", "") or "", getattr(block, "latex", "") or ""
    
    t = _global_clean(t)
    if l:
        # [구조 개선] 원문 보존을 위해 LaTeX 원본 유지 ($ 델리미터 보존)
        l_wrapped = l if l.startswith("$") or l.startswith("\\") else f"${l}$"
        if t:
            # 텍스트와 LaTeX 중복 여부 판단 시 통합 정규화 사용
            t_norm = _normalize_for_compare(t)
            lp_norm = _normalize_for_compare(l)
            
            if t_norm and lp_norm and (lp_norm in t_norm or t_norm in lp_norm):
                return l_wrapped
            return f"{t} {l_wrapped}"
        return l_wrapped
    return t

def _collect_problems(state: AgentState) -> List[str]:
    file_processing = state.get("file_processing")
    if not file_processing: return []
    
    ocr_blocks = file_processing.get("ocr_blocks") if isinstance(file_processing, dict) else getattr(file_processing, "ocr_blocks", None)
    if not ocr_blocks: return []
    
    pages = ocr_blocks.get("pages", []) if isinstance(ocr_blocks, dict) else getattr(ocr_blocks, "pages", [])
    if not pages: return []
    
    problems_map = {}
    problem_seen_norms = {} 
    last_num = None
    
    for page in pages:
        blocks = page.get("blocks", []) if isinstance(page, dict) else getattr(page, "blocks", [])
        
        for block in blocks:
            text = _block_text(block)
            if not text: continue
            
            num = extract_problem_number(text)
            target_num = num if num else last_num
            
            if target_num is not None:
                if target_num not in problems_map:
                    problems_map[target_num] = []
                    problem_seen_norms[target_num] = set()
                
                norm = _normalize_for_compare(text)
                if norm and any(norm in seen or seen in norm for seen in problem_seen_norms[target_num]):
                    continue
                
                problems_map[target_num].append(text)
                problem_seen_norms[target_num].add(norm)
                if num: last_num = num
            else:
                if 0 not in problems_map:
                    problems_map[0] = []
                    problem_seen_norms[0] = set()
                
                norm = _normalize_for_compare(text)
                if norm not in problem_seen_norms[0]:
                    problems_map[0].append(text)
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
        else:
            pass

    sorted_keys = sorted(problems_map.keys())
    return ["\n".join(problems_map[k]).strip() for k in sorted_keys]

def _align_explanations_to_problems(problems, explanations):
    """[구조 개선] 정렬 단계에서는 단순 매핑만 수행"""
    return [str(e).strip() for e in explanations[:len(problems)]]

def _get_problem_title(text: str) -> str:
    """문제 본문에서 첫 줄을 추출하여 요약 제목 생성"""
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
        num = extract_problem_number(p)
        e_clean = _global_clean(e)
        
        ans_match = ANS_RE.search(e_clean)
        ans = ans_match.group(1).strip() if ans_match else "-"
        
        expl = ANS_RE.sub("", e_clean).strip()
        expl = re.sub(r"(?m)^\d+[\.\)]\s*", "", expl)
        expl = re.sub(r"^해설\s*[:：]\s*", "", expl)
        
        entries.append({
            "number": num,
            "title": _get_problem_title(p),
            "original": p.strip(),          # 원본 LaTeX 보존
            "answer": ans,                  # 원본 보존 (graph.py에서 변환)
            "explanation": expl.strip()     # 원본 보존 (graph.py에서 변환)
        })
    return entries
