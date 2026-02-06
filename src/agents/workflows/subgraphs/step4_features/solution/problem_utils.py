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

# 정규식 패턴 강화: 문제 1, (1), Q1, No.1, [1] 등 다양한 형식 지원
PROBLEM_MARKER_PATTERN = re.compile(
    r"(?mi)^\s*(?:문제|Q|No|Task|Step|\[|#)?\s*(\d{1,3})[\.\)\]]?"
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
    # 1. 원문 정규화 및 원문 숫자 변환
    t = unicodedata.normalize("NFKC", text)
    for circle, num in CIRCLED_NUMBER_MAP.items():
        if not str(circle).isdigit():
            t = t.replace(circle, str(num))
    
    # 2. LaTeX 명령어 및 서식 명령 제거
    t = re.sub(r"\\[a-zA-Z]+", "", t) 
    
    # 3. 순수 의미(숫자, 한글, 영문)만 남김
    t = re.sub(r"[^0-9a-zA-Z가-힣]", "", t)
    return t.lower()

def extract_problem_number(text: str) -> Optional[int]:
    """텍스트에서 문제 번호를 추출합니다. (캐싱 고려 가능)"""
    if not text: return None
    clean_t = _global_clean(text)
    
    # 1. 정규식 패턴 매칭 (문제 1, (1), Q1 등)
    match = PROBLEM_MARKER_PATTERN.search(clean_t)
    if match:
        val = match.group(1)
        if val.isdigit():
            return int(val)
            
    # 2. 원문 숫자 (① 등) 매칭
    circle_match = re.search(r"([①-⑳])", clean_t)
    if circle_match:
        return CIRCLED_NUMBER_MAP.get(circle_match.group(1))
        
    return None

def split_text_into_problems(text: str) -> List[str]:
    """텍스트 내의 문제 마커(문제 1, Q2 등)를 기준으로 텍스트를 분할합니다."""
    if not text:
        return []

    # 마커를 기준으로 분할 (마커 자체를 유지하기 위해 캡처 그룹 사용)
    pattern = re.compile(
        r"((?:문제|Q|No|Task|Step|\[|#)\s*\d{1,3}[\.\)\]]?)", re.IGNORECASE | re.MULTILINE
    )
    parts = pattern.split(text)

    problems = []
    # 첫 번째 파트(첫 번째 마커 전의 텍스트) 처리
    first_part = parts[0].strip()
    if first_part:
        # 만약 첫 번째 파트가 너무 짧거나 의미 없는 명령문("문제 풀어줘" 등)이면 제외 고려 가능
        if len(first_part) > 2:
            problems.append(first_part)

    # 마커와 그 뒤의 내용을 합침
    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        combined = (marker + content).strip()
        if combined:
            problems.append(combined)

    return [p for p in problems if p]


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
                # 부분 일치로도 중복 판단 (유사 선지 제거 강화)
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
            # 헤더 미포함 시에도 최소한 데이터는 유지 (0번 키에 그대로 둠)
            pass

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
