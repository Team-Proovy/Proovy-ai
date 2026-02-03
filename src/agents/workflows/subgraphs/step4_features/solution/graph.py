"""Solution feature subgraph.
해설 텍스트는 LLM으로 만들고, PDF는 e2b_runner에서 파이썬 코드로 생성한다.

"""


from __future__ import annotations

import json
import math
import os
import re
from typing import Any, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from agents.state import AgentState, SolutionProgress, SolutionResult
from agents.tools import E2BExecutionError, run_python_with_e2b
from agents.workflows.utils import (
    call_model,
    extract_ocr_text,
    recent_user_context,
    safe_json_loads,
)
from schema.models import OpenRouterModelName


CHUNK_SIZE_DEFAULT = 5
PROBLEM_MARKER_PATTERN = re.compile(r"(?m)^\s*(?:문제\s*)?\d{1,3}[\.\)]\s+")
OPTION_LIKE_PATTERN = re.compile(
    r"^\s*(?:[A-D]|[가-라]|[A-D가-라]형|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|\d+\s*/\s*\d+|\d+)\s*$"
)
LEADING_NUMBER_PATTERN = re.compile(r"^\s*(?:문제\s*)?(\d{1,3})\s*(?:번|[.\)])?")
SOLUTION_JSON_EXAMPLE = '{"explanations":["해설1","해설2"],"chunk_summary":"요약"}'


def _ensure_progress(state: AgentState) -> SolutionProgress:
    progress = state.get("solution_progress")
    if isinstance(progress, dict):
        progress = SolutionProgress(**progress)
    if progress is None:
        progress = SolutionProgress(chunk_size=CHUNK_SIZE_DEFAULT)
    if not progress.chunk_size:
        progress.chunk_size = CHUNK_SIZE_DEFAULT
    state["solution_progress"] = progress
    return progress


def _ensure_solution_result(state: AgentState) -> SolutionResult:
    result = state.get("solution_result")
    if isinstance(result, dict):
        result = SolutionResult(**result)
    if result is None:
        result = SolutionResult()
    state["solution_result"] = result
    return result


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
    return page_blocks


def _split_by_problem_markers(text: str) -> List[str]:
    if not text:
        return []
    matches = list(PROBLEM_MARKER_PATTERN.finditer(text))
    if len(matches) < 2:
        return [text.strip()]
    segments: List[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def _is_probable_problem_start(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
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


def _collect_problems(state: AgentState) -> List[str]:
    page_blocks = _page_blocks_from_state(state)
    problems_with_meta: List[Tuple[int, int, str]] = []
    for page_idx, blocks in enumerate(page_blocks):
        if not blocks:
            continue
        has_marker = any(_is_probable_problem_start(text) for text in blocks)
        if not has_marker:
            page_text = "\n".join(blocks).strip()
            if page_text:
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
                else:
                    current_parts = [stripped]
                    current_meta = (page_idx, block_idx)
        if current_parts:
            problems_with_meta.append(
                (current_meta[0], current_meta[1], "\n".join(current_parts).strip())
            )

    if problems_with_meta:
        if os.getenv("SOLUTION_SORT_BY_NUMBER") == "1":
            extracted: List[Tuple[int, int, int, Optional[int], str]] = []
            for idx, (page_idx, block_idx, text) in enumerate(problems_with_meta):
                match = LEADING_NUMBER_PATTERN.match(text.strip())
                num = int(match.group(1)) if match else None
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


def _build_pdf_code(payload: dict) -> str:
    # e2b 샌드박스에서 실행할 파이썬 코드 문자열을 생성한다.
    payload_json = json.dumps(payload, ensure_ascii=False)
    template = """
import json
import os
import sys
import subprocess
import base64
import urllib.request
import shutil
import socket

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

data = json.loads(__PAYLOAD_JSON__)
pdf_path = data.get("pdf_path") or "/home/user/solution.pdf"
file_name = data.get("file_name") or os.path.basename(pdf_path)
emit_base64 = bool(data.get("emit_base64"))
font_urls = data.get("font_urls") or []
if isinstance(font_urls, str):
    font_urls = [font_urls]
font_base64 = data.get("font_base64")

c = canvas.Canvas(pdf_path, pagesize=A4)
width, height = A4
margin = 48
y = height - margin
font_name = "Helvetica"
font_size = 11
title_size = 15
question_size = 12
answer_size = 10
section_gap = 8
separator_line = "-" * 48
font_path = os.getenv(
    "SOLUTION_FONT_PATH",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
)
font_loaded = False

def download_font(url, dest, timeout=10):
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "SolutionPDFAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return False
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
        os.chmod(dest, 0o644)
        if os.path.getsize(dest) < 1024:
            return False
        return True
    except (urllib.error.URLError, socket.timeout, PermissionError):
        return False

if not os.path.exists(font_path) and font_urls:
    for url in font_urls:
        if not url:
            continue
        if download_font(url, font_path):
            break
if not os.path.exists(font_path) and font_base64:
    try:
        os.makedirs(os.path.dirname(font_path), exist_ok=True)
        with open(font_path, "wb") as out:
            out.write(base64.b64decode(font_base64))
        os.chmod(font_path, 0o644)
    except Exception:
        pass
if os.path.exists(font_path):
    try:
        pdfmetrics.registerFont(TTFont("NotoSansKR", font_path))
        font_name = "NotoSansKR"
        font_loaded = True
    except Exception:
        font_name = "Helvetica"
c.setFont(font_name, font_size)
line_height = font_size + 4

def draw_wrapped(text, *, font=None, size=None, extra_gap=0):
    global y
    use_font = font or font_name
    use_size = size or font_size
    max_width = width - 2 * margin
    lines = simpleSplit(str(text), use_font, use_size, max_width)
    for ln in lines:
        if y < margin + (use_size + 4):
            c.showPage()
            c.setFont(use_font, use_size)
            y = height - margin
        c.setFont(use_font, use_size)
        c.drawString(margin, y, ln)
        y -= (use_size + 4)
    if extra_gap:
        y -= extra_gap

title = data.get("title", "Solution")
draw_wrapped(title, size=title_size, extra_gap=section_gap)
draw_wrapped(separator_line, size=answer_size, extra_gap=section_gap)

entries = data.get("entries", [])
for idx, entry in enumerate(entries, start=1):
    q = entry.get("problem", "")
    a = entry.get("explanation", "")
    draw_wrapped(separator_line, size=answer_size, extra_gap=section_gap)
    draw_wrapped(f"Q{idx}. {q}", size=question_size, extra_gap=section_gap)
    draw_wrapped(str(a), size=answer_size, extra_gap=section_gap)
    draw_wrapped("")

summary = data.get("summary")
if summary:
    draw_wrapped("Summary:", size=question_size, extra_gap=section_gap)
    draw_wrapped(str(summary), size=answer_size)

c.save()
try:
    file_size = os.path.getsize(pdf_path)
except Exception:
    file_size = None
print(f"PDF_PATH: {pdf_path}")
print(f"PDF_NAME: {file_name}")
print(f"PDF_SIZE: {file_size}")
print(f"PDF_FONT: {font_name}")
print(f"PDF_FONT_PATH: {font_path}")
print(f"PDF_FONT_LOADED: {font_loaded}")
if emit_base64:
    try:
        with open(pdf_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        print(f"PDF_BASE64: {encoded}")
    except Exception as exc:
        print(f"PDF_BASE64_ERROR: {exc}")
""".strip()
    return template.replace("__PAYLOAD_JSON__", repr(payload_json))


def _extract_pdf_meta(
    stdout: List[str] | str,
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[bool],
]:
    if isinstance(stdout, str):
        raw_lines = stdout.splitlines()
    else:
        raw_lines = list(stdout)
    lines: List[str] = []
    for item in raw_lines:
        if isinstance(item, str):
            lines.extend(item.splitlines())
        else:
            lines.append(str(item))
    pdf_path = None
    pdf_name = None
    pdf_size: Optional[int] = None
    pdf_base64: Optional[str] = None
    pdf_font: Optional[str] = None
    pdf_font_path: Optional[str] = None
    pdf_font_loaded: Optional[bool] = None
    for line in reversed(lines):
        if "PDF_PATH:" in line:
            pdf_path = line.split("PDF_PATH:", 1)[-1].strip() or None
        if "PDF_NAME:" in line:
            pdf_name = line.split("PDF_NAME:", 1)[-1].strip() or None
        if "PDF_SIZE:" in line:
            raw = line.split("PDF_SIZE:", 1)[-1].strip()
            try:
                pdf_size = int(raw)
            except (TypeError, ValueError):
                pdf_size = None
        if "PDF_BASE64:" in line and pdf_base64 is None:
            pdf_base64 = line.split("PDF_BASE64:", 1)[-1].strip() or None
        if "PDF_FONT:" in line and pdf_font is None:
            pdf_font = line.split("PDF_FONT:", 1)[-1].strip() or None
        if "PDF_FONT_PATH:" in line and pdf_font_path is None:
            pdf_font_path = line.split("PDF_FONT_PATH:", 1)[-1].strip() or None
        if "PDF_FONT_LOADED:" in line and pdf_font_loaded is None:
            raw = line.split("PDF_FONT_LOADED:", 1)[-1].strip()
            pdf_font_loaded = raw.lower() == "true"
        if pdf_path and pdf_name and pdf_size is not None and (
            pdf_base64 is not None or pdf_base64 is None
        ):
            break
    return (
        pdf_path,
        pdf_name,
        pdf_size,
        pdf_base64,
        pdf_font,
        pdf_font_path,
        pdf_font_loaded,
    )


def _build_solution_prompts(problems: List[str]) -> Tuple[str, str]:
    system_prompt = (
        "You are a Korean math tutor. Provide detailed explanations in Korean.\n"
        "Return ONLY valid JSON. No markdown, no extra text.\n"
        "Keys: explanations (list), chunk_summary (string).\n"
        "The length of explanations MUST equal the number of problems and keep order."
    )
    user_prompt = (
        "다음 문제들에 대한 해설을 작성해 주세요.\n"
        "출력은 반드시 JSON만 반환하세요.\n\n"
        "예시 형식:\n"
        f"{SOLUTION_JSON_EXAMPLE}\n\n"
        "문제 목록:\n"
        + json.dumps({"problems": problems}, ensure_ascii=False, indent=2)
    )
    return system_prompt, user_prompt


def _parse_solution_payload(raw: Any) -> Tuple[List[str], Optional[str]]:
    payload = safe_json_loads(raw) if isinstance(raw, str) else None
    if not isinstance(payload, dict):
        payload = {}
    explanations = payload.get("explanations") or []
    if not isinstance(explanations, list):
        explanations = [str(explanations)]
    explanations = [str(item).strip() for item in explanations if str(item).strip()]
    chunk_summary = str(payload.get("chunk_summary") or "").strip() or None
    return explanations, chunk_summary


def _request_explanations(
    problems: List[str],
    tool_outputs: dict,
) -> Tuple[List[str], Optional[str]]:
    system_prompt, user_prompt = _build_solution_prompts(problems)
    raw = call_model(OpenRouterModelName.GPT_5_MINI, system_prompt, user_prompt)
    explanations, chunk_summary = _parse_solution_payload(raw)

    if len(explanations) < len(problems):
        if isinstance(raw, str):
            tool_outputs["solution_llm_raw"] = raw
        retry_raw = call_model(OpenRouterModelName.GPT_5_MINI, system_prompt, user_prompt)
        if isinstance(retry_raw, str):
            tool_outputs["solution_llm_retry_raw"] = retry_raw
        retry_explanations, retry_summary = _parse_solution_payload(retry_raw)
        if len(retry_explanations) >= len(explanations):
            explanations = retry_explanations
            if retry_summary:
                chunk_summary = retry_summary

    if len(explanations) < len(problems):
        explanations.extend(["해설을 생성하지 못했습니다."] * (len(problems) - len(explanations)))

    return explanations, chunk_summary


def solution(state: AgentState) -> AgentState:
    """해설 기능 (청크 기반)."""
    print("---FEATURE: SOLUTION---")

    progress = _ensure_progress(state)
    solution_result = _ensure_solution_result(state)
    problems = state.get("solution_chunks") or _collect_problems(state)
    state["solution_chunks"] = problems

    total_problems = len(problems)
    chunk_size = int(progress.chunk_size or CHUNK_SIZE_DEFAULT)
    total_chunks = math.ceil(total_problems / chunk_size) if total_problems else 0

    progress.total_problems = total_problems
    progress.total_chunks = total_chunks

    if total_chunks == 0:
        solution_result.guide = "해설할 문제가 없습니다."
        state["solution_result"] = solution_result
        state["prev_action"] = "Solution"
        return state
    if progress.done:
        solution_result.guide = solution_result.guide or "모든 청크가 이미 처리되었습니다."
        state["solution_result"] = solution_result
        state["prev_action"] = "Solution"
        return state

    current_chunk_index = max(0, min(int(progress.current_chunk or 0), total_chunks - 1))
    start = current_chunk_index * chunk_size
    end = min(start + chunk_size, total_problems)
    chunk_problems = problems[start:end]

    tool_outputs = state.setdefault("tool_outputs", {})
    explanations, chunk_summary = _request_explanations(chunk_problems, tool_outputs)

    solution_result.guide = chunk_summary or solution_result.guide or "해설을 생성했습니다."
    solution_result.chunk_index = current_chunk_index + 1
    solution_result.chunk_size = chunk_size
    solution_result.total_problems = total_problems
    solution_result.total_chunks = total_chunks
    solution_result.problems = chunk_problems
    solution_result.explanations = explanations
    solution_result.chunk_summary = chunk_summary

    pdf_file_name = f"solution_chunk_{solution_result.chunk_index}.pdf"
    emit_base64 = os.getenv("SOLUTION_EMIT_PDF_BASE64") == "1"
    font_urls_env = os.getenv("SOLUTION_FONT_URLS")
    if font_urls_env:
        font_urls = [item.strip() for item in font_urls_env.split(",") if item.strip()]
    else:
        font_url = os.getenv("SOLUTION_FONT_URL")
        font_urls = (
            [font_url]
            if font_url
            else [
                "https://cdn.jsdelivr.net/gh/google/fonts/ofl/notosanskr/NotoSansKR-Regular.ttf",
                "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR-Regular.ttf",
            ]
        )
    font_base64 = os.getenv("SOLUTION_FONT_BASE64")
    if not font_base64:
        font_base64_path = os.getenv("SOLUTION_FONT_BASE64_PATH")
        if font_base64_path:
            try:
                with open(font_base64_path, "r", encoding="utf-8") as handle:
                    font_base64 = re.sub(r"\s+", "", handle.read())
            except OSError:
                font_base64 = None
    pdf_payload = {
        "title": f"Solution Chunk {solution_result.chunk_index}/{total_chunks}",
        "entries": [
            {"problem": prob, "explanation": exp}
            for prob, exp in zip(chunk_problems, explanations)
        ],
        "summary": chunk_summary,
        "pdf_path": f"/home/user/{pdf_file_name}",
        "file_name": pdf_file_name,
        "emit_base64": emit_base64,
        "font_urls": font_urls,
        "font_base64": font_base64,
    }
    pdf_code = _build_pdf_code(pdf_payload)
    try:
        execution = run_python_with_e2b(pdf_code)

        raw_stdout = execution.stdout if hasattr(execution, "stdout") else execution
        if isinstance(raw_stdout, str):
            stdout_lines = raw_stdout.splitlines()
        elif isinstance(raw_stdout, list):
            stdout_lines = raw_stdout
        else:
            stdout_lines = [str(raw_stdout)]

        raw_stderr = execution.stderr if hasattr(execution, "stderr") else None
        if isinstance(raw_stderr, str):
            stderr_lines = raw_stderr.splitlines()
        elif isinstance(raw_stderr, list):
            stderr_lines = raw_stderr
        else:
            stderr_lines = [] if raw_stderr is None else [str(raw_stderr)]

        (
            pdf_path,
            pdf_name,
            pdf_size,
            pdf_base64,
            pdf_font,
            pdf_font_path,
            pdf_font_loaded,
        ) = _extract_pdf_meta(stdout_lines)
        pdf_path = pdf_path or pdf_payload["pdf_path"]
        pdf_name = pdf_name or pdf_file_name
        solution_result.pdf_path = pdf_path
        solution_result.pdf_file_name = pdf_name
        solution_result.pdf_mime_type = "application/pdf"
        solution_result.pdf_file_size = pdf_size
        solution_result.pdf_error = None
        tool_outputs["solution_pdf"] = {
            "success": True,
            "stdout": stdout_lines,
            "stderr": stderr_lines,
            "pdf_path": pdf_path,
            "file_name": pdf_name,
            "mime_type": "application/pdf",
            "file_size": pdf_size,
            "pdf_base64": pdf_base64,
            "pdf_font": pdf_font,
            "pdf_font_path": pdf_font_path,
            "pdf_font_loaded": pdf_font_loaded,
        }
    except Exception as exc:
        solution_result.pdf_error = str(exc)
        tool_outputs["solution_pdf"] = {
            "success": False,
            "stdout": [],
            "stderr": [str(exc)],
        }

    progress.current_chunk = current_chunk_index + 1
    progress.done = progress.current_chunk >= total_chunks
    remaining_count = max(0, total_problems - (progress.current_chunk * chunk_size))
    state["solution_progress"] = progress

    final_output = state.setdefault("final_output", {})
    final_output["solution"] = {
        "chunk_index": solution_result.chunk_index,
        "chunk_size": solution_result.chunk_size,
        "total_problems": solution_result.total_problems,
        "total_chunks": solution_result.total_chunks,
        "problems": solution_result.problems,
        "explanations": solution_result.explanations,
        "chunk_summary": solution_result.chunk_summary,
        "pdf_path": solution_result.pdf_path,
        "pdf_file_name": solution_result.pdf_file_name,
        "pdf_mime_type": solution_result.pdf_mime_type,
        "pdf_file_size": solution_result.pdf_file_size,
        "pdf_error": solution_result.pdf_error,
        "next_chunk_available": not progress.done,
        "remaining_count": remaining_count,
    }
    if tool_outputs.get("solution_pdf"):
        final_output["solution"]["pdf_font"] = tool_outputs["solution_pdf"].get(
            "pdf_font"
        )
        final_output["solution"]["pdf_font_path"] = tool_outputs["solution_pdf"].get(
            "pdf_font_path"
        )
        final_output["solution"]["pdf_font_loaded"] = tool_outputs["solution_pdf"].get(
            "pdf_font_loaded"
        )
    if emit_base64:
        final_output["solution"]["pdf_base64"] = tool_outputs.get("solution_pdf", {}).get(
            "pdf_base64"
        )
    if not progress.done:
        next_batch = min(chunk_size, remaining_count) if remaining_count else chunk_size
        final_output["solution"]["suggestions"] = [
            f"다음 {next_batch}문제도 풀어드릴까요?",
            "전체 요약본이 필요하신가요?",
        ]
    state["final_output"] = final_output
    state["solution_result"] = solution_result
    state["prev_action"] = "Solution"
    return state


builder = StateGraph(AgentState)
builder.add_node("Solution", solution)
builder.set_entry_point("Solution")
builder.add_edge("Solution", END)

graph = builder.compile()
