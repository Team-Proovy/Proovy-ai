from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_CIRCLED_MAP = {ch: idx + 1 for idx, ch in enumerate(_CIRCLED_NUMBERS)}

_PATTERNS = (
    re.compile(r"^\s*(?:문제\s*)?(\d{1,3})\s*[\.\)](?:\s|$)"),
    re.compile(r"^\s*(?:문제\s*)?(\d{1,3})\s*번(?:\s|$)"),
    re.compile(r"^\s*[\(\[]\s*(\d{1,3})\s*[\)\]](?:\s|$)"),
    re.compile(r"^\s*([①-⑳])(?:\s|$)"),
)


def _extract_path_from_file_item(item: Any) -> Optional[str]:
    if not item:
        return None
    if isinstance(item, (str, Path)):
        return str(item)
    if isinstance(item, dict):
        for key in ("path", "file_path", "url", "s3_uri", "filename", "name", "file_name"):
            value = item.get(key)
            if value:
                return str(value)
    return None


def input_files_fingerprint(state: Dict[str, Any]) -> str:
    """input_files 목록의 경로 기반 fingerprint를 반환한다."""
    raw_files = state.get("input_files") or []
    if isinstance(raw_files, (str, Path)):
        raw_files = [raw_files]
    paths: List[str] = []
    if isinstance(raw_files, list):
        for item in raw_files:
            extracted = _extract_path_from_file_item(item)
            if extracted:
                paths.append(extracted)
    if not paths:
        return ""
    canonical = "|".join(sorted(set(paths)))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _iter_ocr_block_lines(state: Dict[str, Any]) -> List[str]:
    file_processing = state.get("file_processing")
    if not file_processing:
        return []
    if isinstance(file_processing, dict):
        ocr_blocks = file_processing.get("ocr_blocks")
    else:
        ocr_blocks = getattr(file_processing, "ocr_blocks", None)
    if not ocr_blocks:
        return []

    pages = ocr_blocks.get("pages") if isinstance(ocr_blocks, dict) else getattr(ocr_blocks, "pages", None)
    if not isinstance(pages, list):
        return []

    lines: List[str] = []
    for page in pages:
        blocks = page.get("blocks") if isinstance(page, dict) else getattr(page, "blocks", None)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict):
                text = str(block.get("text") or "").strip()
                latex = str(block.get("latex") or "").strip()
            else:
                text = str(getattr(block, "text", "") or "").strip()
                latex = str(getattr(block, "latex", "") or "").strip()

            merged = text
            if latex and latex not in merged:
                merged = f"{merged} {latex}".strip() if merged else latex
            if merged:
                lines.append(merged)
    return lines


def _extract_problem_number(line: str) -> Tuple[Optional[int], str]:
    if not line:
        return None, ""
    for pattern in _PATTERNS:
        match = pattern.match(line)
        if not match:
            continue
        raw = match.group(1)
        if raw in _CIRCLED_MAP:
            return _CIRCLED_MAP[raw], raw
        if raw.isdigit():
            return int(raw), raw
    return None, ""


def _looks_like_choice_line(line: str) -> bool:
    """선지(예: ① A형, ② 3/4)처럼 보이는 짧은 라인은 문제 시작으로 보지 않는다."""
    compact = re.sub(r"\s+", "", line or "")
    if not compact:
        return False
    if compact[0] not in _CIRCLED_MAP:
        return False
    # 짧은 선택지 라인: ①A형, ②B형, ③9/4, ④ㄱ·ㄴ 등
    if len(compact) <= 16:
        return True
    # 보기/선택지 라벨이 포함된 경우도 선지로 취급
    if any(token in compact for token in ("보기", "선지", "선택지")):
        return True
    return False


def extract_problem_inventory(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """OCR 블록을 문제 단위로 분할해 문제 인벤토리로 반환한다."""
    lines = _iter_ocr_block_lines(state)
    if not lines:
        return []

    inventory: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    header_lines: List[str] = []

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        number, marker = _extract_problem_number(line)
        if number is not None:
            # 원형 숫자(①~⑳)는 선택지 오인식이 잦아, 짧은 라인은 문제 시작에서 제외한다.
            if marker in _CIRCLED_MAP and _looks_like_choice_line(line):
                if current is None:
                    header_lines.append(line)
                else:
                    current["text"] = f"{current['text']}\n{line}".strip()
                continue

            if current and current.get("text"):
                inventory.append(current)
            
            current = {"number": number, "text": line, "marker": marker}
            if not inventory and header_lines:
                # 첫 번째 번호 붙은 문제 앞에 있던 텍스트(헤더 등)를 본문에 합쳐준다.
                header_text = "\n".join(header_lines)
                current["text"] = f"{header_text}\n{current['text']}".strip()
                header_lines = []
            continue

        if current is None:
            header_lines.append(line)
        else:
            current["text"] = f"{current['text']}\n{line}".strip()

    if current and current.get("text"):
        inventory.append(current)
    elif not inventory and header_lines:
        # 번호 있는 문제가 하나도 없었던 경우에만 전체를 하나의 문제로 간주한다.
        inventory.append({"number": None, "text": "\n".join(header_lines), "marker": ""})

    return inventory

