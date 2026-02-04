from __future__ import annotations

import base64
import io
import json
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List, Optional, Tuple


def _build_pdf_code(payload: dict) -> str:
    # e2b 샌드박스에서 실행할 파이썬 코드 문자열을 생성한다.
    payload_json = json.dumps(payload, ensure_ascii=False)
    payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("ascii")
    template = """
import json
import os
import base64
import io
import re
import urllib.request
import shutil
import socket
import sys
import subprocess

render_latex = os.getenv("SOLUTION_RENDER_LATEX", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

_DEPS_INSTALLED = False


def _maybe_install_deps():
    global _DEPS_INSTALLED
    if _DEPS_INSTALLED:
        return True
    flag = os.getenv("SOLUTION_E2B_INSTALL_DEPS", "").strip().lower()
    if flag not in ("1", "true", "yes", "y", "on"):
        return False
    print("DEPS_INSTALL_START")
    try:
        packages = ["reportlab"]
        if render_latex:
            packages.append("matplotlib")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                *packages,
            ]
        )
        _DEPS_INSTALLED = True
        print("DEPS_INSTALL_DONE")
        return True
    except Exception as exc:
        print(f"DEPS_INSTALL_FAIL: {exc}")
        return False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit, ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except Exception:
    if _maybe_install_deps():
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import simpleSplit, ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    else:
        raise
plt = None
if render_latex:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:
        if _maybe_install_deps():
            try:
                import matplotlib
                matplotlib.use("Agg")
                from matplotlib import pyplot as plt
            except Exception:
                plt = None

data = json.loads(base64.b64decode("__PAYLOAD_JSON_B64__").decode("utf-8"))
pdf_path = data.get("pdf_path") or "/home/user/solution.pdf"
file_name = data.get("file_name") or os.path.basename(pdf_path)
emit_base64 = bool(data.get("emit_base64"))
font_urls = data.get("font_urls") or []
if isinstance(font_urls, str):
    font_urls = [font_urls]
font_base64 = data.get("font_base64")
print(f"ENV_RENDER_LATEX: {render_latex}")
print(f"ENV_INSTALL_DEPS: {os.getenv('SOLUTION_E2B_INSTALL_DEPS') or ''}")

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
math_dpi = 200
latex_line_pattern = re.compile(r"\\[a-zA-Z]+")
option_line_pattern = re.compile(
    r"^\s*(?:[①-⑳]|[ㄱ-ㅎ]\.|[A-D]\.|[가-라]\.|\\d{1,3}[.)])\s+"
)
hangul_pattern = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
latex_token_replacements = {
    "\\\\times": "×",
    "\\\\cdot": "·",
    "\\\\pi": "π",
    "\\\\infty": "∞",
    "\\\\sum": "∑",
    "\\\\ln": "ln",
    "\\\\to": "→",
    "\\\\le": "≤",
    "\\\\ge": "≥",
    "\\\\neq": "≠",
    "\\\\approx": "≈",
    "\\\\sim": "~",
}


def _normalize_latex(text):
    cleaned = text.strip()
    cleaned = re.sub(r"\\text\\{([^}]*)\\}", r"\\1", cleaned)
    return cleaned


def _replace_latex_tokens(text):
    cleaned = re.sub(r"\\\\text\\{([^}]*)\\}", r"\\1", str(text))
    cleaned = re.sub(
        r"\\\\frac\\{([^{}]+)\\}\\{([^{}]+)\\}",
        r"(\\1)/(\\2)",
        cleaned,
    )
    for key, value in latex_token_replacements.items():
        cleaned = cleaned.replace(key, value)
    return cleaned


def _contains_hangul(text):
    return bool(hangul_pattern.search(text))


def _is_latex_line(text):
    if not text:
        return False
    if _contains_hangul(text):
        return False
    if "\\\\" in text or "$" in text:
        return True
    if any(sym in text for sym in ("∑", "Σ", "√", "^", "_")):
        return True
    token_count = len(latex_line_pattern.findall(text))
    return token_count >= 1


def _render_latex_image(text, size):
    if plt is None:
        return None
    latex = _normalize_latex(text)
    if not latex:
        return None
    if not latex.startswith("$"):
        latex = f"${latex}$"
    fig = None
    try:
        fig = plt.figure()
        fig.text(0, 0, latex, fontsize=size)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=math_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=True,
        )
        plt.close(fig)
        buffer.seek(0)
        return buffer
    except Exception:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
        return None


def _is_option_like_line(text):
    if not text:
        return False
    return bool(option_line_pattern.match(text))


def _normalize_segments(text):
    lines = [ln.rstrip() for ln in str(text).splitlines()]
    segments = []
    buffer = []

    def flush_buffer():
        if not buffer:
            return
        paragraph = " ".join(part.strip() for part in buffer if part.strip())
        paragraph = _replace_latex_tokens(paragraph)
        if paragraph:
            segments.append({"type": "text", "text": paragraph})
        buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_buffer()
            segments.append({"type": "blank"})
            continue
        if _is_latex_line(stripped):
            flush_buffer()
            segments.append({"type": "latex", "text": stripped})
            continue
        if stripped.startswith("정답:"):
            flush_buffer()
            segments.append({"type": "text", "text": _replace_latex_tokens(stripped)})
            segments.append({"type": "blank"})
            continue
        if stripped in {"보기", "보기:"} or _is_option_like_line(stripped):
            flush_buffer()
            segments.append({"type": "text", "text": _replace_latex_tokens(stripped)})
            continue
        buffer.append(stripped)

    flush_buffer()
    return segments

def draw_wrapped(text, *, font=None, size=None, extra_gap=0):
    global y
    use_font = font or font_name
    use_size = size or font_size
    max_width = width - 2 * margin
    for segment in _normalize_segments(text):
        seg_type = segment["type"]
        if seg_type == "blank":
            y -= (use_size + 4)
            continue
        if seg_type == "latex":
            image_buf = _render_latex_image(segment["text"], use_size + 2)
            if image_buf is None:
                continue
            img = ImageReader(image_buf)
            img_w, img_h = img.getSize()
            width_pt = img_w * 72 / math_dpi
            height_pt = img_h * 72 / math_dpi
            if width_pt > max_width:
                scale = max_width / width_pt
                width_pt *= scale
                height_pt *= scale
            if y < margin + height_pt:
                c.showPage()
                c.setFont(use_font, use_size)
                y = height - margin
            c.drawImage(
                img,
                margin,
                y - height_pt,
                width=width_pt,
                height=height_pt,
                mask="auto",
            )
            y -= (height_pt + 4)
            continue
        wrapped = simpleSplit(segment["text"], use_font, use_size, max_width)
        for ln in wrapped:
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
for entry in entries:
    label = entry.get("number")
    q = entry.get("problem", "")
    a = entry.get("explanation", "")
    ans = entry.get("answer", "")
    if label is not None:
        question_label = f"{label}. {q}".strip()
    else:
        question_label = str(q)
    draw_wrapped(separator_line, size=answer_size, extra_gap=section_gap)
    draw_wrapped(question_label, size=question_size, extra_gap=section_gap)
    draw_wrapped("정답:", size=answer_size)
    if ans:
        draw_wrapped(str(ans), size=answer_size, extra_gap=section_gap)
    else:
        draw_wrapped("-", size=answer_size, extra_gap=section_gap)
    draw_wrapped(str(a), size=answer_size, extra_gap=section_gap)
    draw_wrapped("")

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
        chunk_size = int(os.getenv("SOLUTION_PDF_BASE64_CHUNK", "4096"))
        if chunk_size < 256:
            chunk_size = 256
        print("PDF_BASE64_BEGIN")
        for i in range(0, len(encoded), chunk_size):
            print(f"PDF_BASE64_CHUNK:{encoded[i:i+chunk_size]}")
        print("PDF_BASE64_END")
    except Exception as exc:
        print(f"PDF_BASE64_ERROR: {exc}")
""".strip()
    return template.replace("__PAYLOAD_JSON_B64__", payload_b64)


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
    base64_chunks: List[str] = []
    collecting = False
    for line in lines:
        if line.startswith("PDF_BASE64:"):
            pdf_base64 = line.split("PDF_BASE64:", 1)[-1].strip() or None
            collecting = False
            base64_chunks = []
            continue
        if line.startswith("PDF_BASE64_BEGIN"):
            collecting = True
            base64_chunks = []
            continue
        if line.startswith("PDF_BASE64_END"):
            if base64_chunks:
                pdf_base64 = "".join(base64_chunks)
            collecting = False
            continue
        if line.startswith("PDF_BASE64_CHUNK:") and collecting:
            chunk = line.split("PDF_BASE64_CHUNK:", 1)[-1].strip()
            if chunk:
                base64_chunks.append(chunk)

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
        if pdf_path and pdf_name and pdf_size is not None:
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


@contextmanager
def _temporary_env(envs: dict[str, str] | None):
    if not envs:
        yield
        return
    previous: dict[str, Optional[str]] = {}
    for key, value in envs.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_local_pdf_path(file_name: str) -> str:
    base_dir = os.getenv("SOLUTION_LOCAL_OUTPUT_DIR", "").strip()
    if not base_dir:
        base_dir = os.path.join("outputs", "solution")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.abspath(os.path.join(base_dir, file_name))


def _merge_local_font_env(
    envs: dict[str, str] | None,
    *,
    pdf_path: str,
) -> dict[str, str] | None:
    if envs is None:
        envs = {}
    if envs.get("SOLUTION_FONT_PATH") or os.getenv("SOLUTION_FONT_PATH"):
        return envs
    font_dir = os.path.dirname(pdf_path) or os.getcwd()
    font_path = os.path.join(font_dir, "NotoSansKR-Regular.ttf")
    merged = dict(envs)
    merged["SOLUTION_FONT_PATH"] = font_path
    return merged


def _execute_pdf_locally(
    pdf_code: str,
    envs: dict[str, str] | None = None,
) -> Tuple[bool, List[str], List[str], Optional[Exception]]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with _temporary_env(envs), redirect_stdout(stdout_buffer), redirect_stderr(
            stderr_buffer
        ):
            exec(pdf_code, {"__name__": "__main__"})
        stdout_lines = stdout_buffer.getvalue().splitlines()
        stderr_lines = stderr_buffer.getvalue().splitlines()
        return True, stdout_lines, stderr_lines, None
    except Exception as exc:
        if stderr_buffer.tell():
            stderr_buffer.write("\n")
        stderr_buffer.write(f"{type(exc).__name__}: {exc}")
        stdout_lines = stdout_buffer.getvalue().splitlines()
        stderr_lines = stderr_buffer.getvalue().splitlines()
        return False, stdout_lines, stderr_lines, exc

