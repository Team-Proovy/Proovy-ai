import json
import os
import re
import base64
import subprocess
import tempfile
import sys
import traceback
import inspect
from typing import Any, Dict, List, Optional, Tuple

# --- [Single Source of Truth] LaTeX to Plain Text Conversion Logic ---
def _convert_latex_block(text: str) -> str:
    """단일 LaTeX 블록을 유니코드/플레인 텍스트로 변환"""
    if not text: return ""
    
    t = text.replace('\\\\', '\\').strip()
    if t.startswith('$$') and t.endswith('$$'):
        t = t[2:-2]
    elif t.startswith('$') and t.endswith('$'):
        t = t[1:-1]
    elif t.startswith(r'\(') and t.endswith(r'\)'): t = t[2:-2]
    elif t.startswith(r'\[') and t.endswith(r'\]'): t = t[2:-2]
    
    # 2. 구조적 변환 (AST-like handling for operators)
    t = re.sub(
        r'\\sum(?:_\{?([^}]*)\}?)?(?:\^\{?([^}]*)\}?)?',
        lambda m: f"Σ({m.group(1) or ''}→{m.group(2) or ''})",
        t,
    )
    t = re.sub(
        r'\\int(?:_\{?([^}]*)\}?)?(?:\^\{?([^}]*)\}?)?',
        lambda m: f"∫({m.group(1) or ''}→{m.group(2) or ''})",
        t,
    )
    t = re.sub(
        r'\\lim_\{?([^}]*)\}?',
        lambda m: "lim(" + m.group(1).replace('\\to', '→') + ")",
        t,
    )
    t = re.sub(r'\\(?:d|t)?frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', t)
    t = re.sub(
        r'\\sqrt(?:\[([^\]]*)\])?\{([^}]*)\}',
        lambda m: f"{m.group(1) or ''}√{m.group(2)}",
        t,
    )
    
    # 3. 그리스 문자 및 특수 기호 매핑
    mapping = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ', r'\epsilon': 'ε',
        r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ', r'\iota': 'ι', r'\kappa': 'κ',
        r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π',
        r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ', r'\phi': 'φ', r'\chi': 'χ',
        r'\psi': 'ψ', r'\omega': 'ω', r'\infty': '∞', r'\to': '→', r'\times': '×',
        r'\cdot': '·', r'\pm': '±', r'\mp': '∓', r'\neq': '≠', r'\le': '≤',
        r'\ge': '≥', r'\partial': '∂', r'\nabla': '∇', r'\forall': '∀', r'\exists': '∃',
        r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\supset': '⊃', r'\cup': '∪',
        r'\cap': '∩', r'\approx': '≈', r'\equiv': '≡', r'\sin': 'sin', r'\cos': 'cos',
        r'\tan': 'tan', r'\exp': 'exp', r'\ln': 'ln', r'\log': 'log',
    }
    for k, v in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        t = t.replace(k, v)
        
    # 4. 지수 및 첨자 (유니코드 변환)
    sup_map = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
    sub_map = str.maketrans("0123456789+-=()n", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ")
    t = re.sub(r'\^\{?([0-9+\-=()n]+)\}?', lambda m: m.group(1).translate(sup_map), t)
    t = re.sub(r'_\{?([0-9+\-=()n]+)\}?', lambda m: m.group(1).translate(sub_map), t)
    
    # 5. 남은 LaTeX 명령어 및 서식 제거
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    t = t.replace('{', '').replace('}', '').replace(r'\(', '').replace(r'\)', '')
    
    return t.strip()

def latex_to_unicode_shared(text: str) -> str:
    """텍스트 내의 모든 LaTeX 블록($...$)을 찾아 변환"""
    if not text: return ""
    
    # 수식 패턴: $$...$$, $...$, \(...\), \[...\]
    math_re = re.compile(r'(\$\$[\s\S]+?\$\$|\$[^$]+\$|\\\(.*?\\\)|\\\[.*?\\\])', re.DOTALL)
    
    result = []
    last_idx = 0
    for match in math_re.finditer(text):
        # 수식 이전의 일반 텍스트 추가
        result.append(text[last_idx:match.start()])
        # 수식 블록 변환하여 추가
        result.append(_convert_latex_block(match.group(0)))
        last_idx = match.end()
    
    # 남은 텍스트 추가
    result.append(text[last_idx:])

    return "".join(result).strip()

# --- PDF Generation Template (E2B Sandbox) ---
template = r"""
import os
import re
import json
import base64
import sys
import traceback
import subprocess
import tempfile
import shutil
from io import BytesIO

# 1. LaTeX to Unicode Shared Logic
{{LATEX_TO_UNICODE_SHARED}}

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader, simpleSplit
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # 2. Configuration
    def _env_truthy(name, default="0"):
        raw = os.getenv(name, default)
        if raw is None:
            return False
        normalized = raw.strip().lower().rstrip(".")
        return normalized in {"1", "true", "yes", "y", "on"}

    data = json.loads(base64.b64decode("{{PAYLOAD}}").decode("utf-8"))
    emit_base64 = bool(data.get("emit_base64"))
    use_math_render = _env_truthy("SOLUTION_USE_MATH_RENDER", "1")
    render_math_as_plain = _env_truthy("SOLUTION_RENDER_MATH_AS_PLAIN", "0")
    complexity_threshold = int(os.getenv("SOLUTION_MATH_COMPLEXITY_THRESHOLD", "3"))
    include_original = _env_truthy("SOLUTION_INCLUDE_ORIGINAL", "1")
    use_svg_render = _env_truthy("SOLUTION_USE_SVG_RENDER", "1")

    font_name = "Helvetica" 
    font_path = "/tmp/font.ttf"
    font_loaded = False

    if data.get("font_base64"):
        with open(font_path, "wb") as f:
            f.write(base64.b64decode(data["font_base64"]))
        pdfmetrics.registerFont(TTFont("NotoSansKR", font_path))
        font_name = "NotoSansKR"
        font_loaded = True
        fe = font_manager.FontEntry(fname=font_path, name="NotoSansKR")
        font_manager.fontManager.ttflist.insert(0, fe)
        plt.rcParams["font.family"] = fe.name
        plt.rcParams['mathtext.fontset'] = 'stix'

    def _ensure_svg_deps():
        try:
            import svglib  # noqa: F401
            return True
        except Exception:
            if not _env_truthy("SOLUTION_SVG_INSTALL_DEPS", "0"):
                return False
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "svglib>=1.5.1", "lxml>=4.9"],
                    check=True,
                    capture_output=True,
                )
                return True
            except Exception:
                return False

    _ensure_svg_deps()

    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        HAS_SVGLIB = True
    except Exception:
        HAS_SVGLIB = False

    try:
        import cairosvg
        HAS_CAIROSVG = True
    except Exception:
        HAS_CAIROSVG = False

    def _clean(t):
        if not t: return ""
        # Preserve paragraph breaks while cleaning
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        # 1. OCR noise or unsupported characters that cause layout gaps
        t = re.sub(r'[\uFFFD\xa0]', '', t)
        # 2. Choice markers (often misinterpreted by OCR as squares)
        t = re.sub(r'[■□▢▣▤▥▦▧▨▩]', '', t)
        # 3. Collapse spaces (keep newlines)
        t = re.sub(r'[ \t]+', ' ', t)
        t = re.sub(r'\n{3,}', '\n\n', t)
        return t.strip()

    def _normalize_math_inner(math_text):
        if not math_text: return ""
        inner = math_text.replace('\\\\', '\\').strip()
        if inner.startswith('$$') and inner.endswith('$$'): inner = inner[2:-2]
        elif inner.startswith('$') and inner.endswith('$'): inner = inner[1:-1]
        elif inner.startswith(r'\(') and inner.endswith(r'\)'): inner = inner[2:-2]
        elif inner.startswith(r'\[') and inner.endswith(r'\]'): inner = inner[2:-2]
        
        inner = re.sub(r'[□■¤]', '', inner)
        inner = re.sub(r'\\boxed\{([^}]*)\}', r'\1', inner)
        for cmd in ['\\displaystyle', '\\textstyle', '\\scriptstyle', '\\scriptscriptstyle', '\\left', '\\right', '\\limits', '\\nolimits']:
            inner = inner.replace(cmd, ' ')
        
        inner = re.sub(r'\\begin\{array\}.*?\\end\{array\}', ' [array] ', inner, flags=re.DOTALL)
        inner = re.sub(r'\\text\{([^}]*)\}', lambda m: m.group(1), inner)
        inner = re.sub(r'([가-힣\s]+)', lambda m: f"\\text{{{m.group(0).strip()}}}", inner)
        return inner.strip()

    def _split_segments(text):
        if not text: return []
        math_re = re.compile(r'(\$\$[\s\S]+?\$\$|\$[^$]+\$|\\\(.*?\\\)|\\\[.*?\\\])', re.DOTALL)
        parts = []
        idx = 0
        for m in math_re.finditer(text):
            if m.start() > idx: parts.append(("text", text[idx:m.start()]))
            parts.append(("math", m.group(0)))
            idx = m.end()
        if idx < len(text): parts.append(("text", text[idx:]))
        return parts

    def _split_math_text_segments(math_text):
        inner = _normalize_math_inner(math_text)
        parts = []
        idx = 0
        for m in re.finditer(r'\\text\{([^}]*)\}', inner):
            if m.start() > idx:
                chunk = inner[idx:m.start()].strip()
                if chunk:
                    parts.append(("math", f"${chunk}$"))
            text_chunk = (m.group(1) or "").strip()
            if text_chunk:
                parts.append(("text", text_chunk))
            idx = m.end()
        
        if idx < len(inner):
            tail = inner[idx:].strip()
            if tail:
                parts.append(("math", f"${tail}$"))
        if not parts:
            return [("math", f"${inner}$")]
        return parts

    _MATH_IMG_CACHE = {}
    _MATH_SVG_CACHE = {}

    _MATHJAX_SCRIPT = (
        "const fs = require('fs');\n"
        "const {mathjax} = require('mathjax-full/js/mathjax.js');\n"
        "const {TeX} = require('mathjax-full/js/input/tex.js');\n"
        "const {SVG} = require('mathjax-full/js/output/svg.js');\n"
        "const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');\n"
        "const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');\n"
        "\n"
        "const input = fs.readFileSync(0, 'utf8') || '{}';\n"
        "const payload = JSON.parse(input);\n"
        "const latex = payload.latex || '';\n"
        "const display = !!payload.display;\n"
        "\n"
        "const adaptor = liteAdaptor();\n"
        "RegisterHTMLHandler(adaptor);\n"
        "const tex = new TeX({packages: ['base','ams','newcommand','bbox']});\n"
        "const svg = new SVG({fontCache: 'none'});\n"
        "const html = mathjax.document('', {InputJax: tex, OutputJax: svg});\n"
        "const node = html.convert(latex, {display: display});\n"
        "const svgOutput = adaptor.outerHTML(node);\n"
        "const bbox = node.getBBox();\n"
        "const result = {svg: svgOutput, width: bbox.w, height: bbox.h, depth: bbox.d};\n"
        "process.stdout.write(JSON.stringify(result));\n"
    )

    def _ensure_mathjax():
        if not use_svg_render:
            return None, None
        if not shutil.which("node"):
            return None, None
        workdir = "/tmp/mathjax"
        os.makedirs(workdir, exist_ok=True)
        script_path = os.path.join(workdir, "mathjax_svg.js")
        if not os.path.exists(script_path):
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(_MATHJAX_SCRIPT)
        node_mod_path = os.path.join(workdir, "node_modules", "mathjax-full")
        if not os.path.exists(node_mod_path):
            if not _env_truthy("SOLUTION_E2B_INSTALL_DEPS", "0") and not _env_truthy("SOLUTION_SVG_INSTALL_DEPS", "0"):
                return None, None
            try:
                subprocess.run(["npm", "init", "-y"], cwd=workdir, check=True, capture_output=True)
                subprocess.run(["npm", "install", "mathjax-full@3"], cwd=workdir, check=True, capture_output=True)
            except Exception:
                return None, None
        return script_path, workdir

    def _render_math_svg_cached(math_text, size):
        if math_text in _MATH_SVG_CACHE: return _MATH_SVG_CACHE[math_text]
        inner = _normalize_math_inner(math_text)
        if not inner: return None
        script_path, workdir = _ensure_mathjax()
        if not script_path:
            return None

        display = False
        if math_text.strip().startswith(r"\[") or math_text.strip().startswith("$$"):
            display = True
        if re.search(r"\\begin\{|\n|\\\\", inner):
            display = True

        try:
            payload = json.dumps({"latex": inner, "display": display}, ensure_ascii=False)
            result = subprocess.run(
                ["node", script_path],
                input=payload,
                text=True,
                capture_output=True,
                cwd=workdir,
                check=True,
            )
            data = json.loads(result.stdout or "{}")
            svg_text = data.get("svg")
            if not svg_text:
                return None
            # MathJax bbox is in em units
            w_pt = float(data.get("width") or 0) * size
            h_pt = float(data.get("height") or 0) * size
            d_pt = float(data.get("depth") or 0) * size
            res = (svg_text, w_pt, h_pt, d_pt)
            _MATH_SVG_CACHE[math_text] = res
            return res
        except Exception as e:
            safe_inner = (inner or "").encode("ascii", "backslashreplace").decode("ascii")
            print(f"SVG_RENDER_ERROR: {e} for content: {safe_inner}")
            return None

    def _render_math_img_cached(math_text, size):
        if math_text in _MATH_IMG_CACHE: return _MATH_IMG_CACHE[math_text]
        inner = _normalize_math_inner(math_text)
        if not inner: return None
        
        try:
            from matplotlib.mathtext import MathTextParser
            parser = MathTextParser('path')
            # Use FontProperties object to avoid 'unhashable dict' error in some matplotlib versions
            prop = font_manager.FontProperties(size=size)
            width, height, descent, glyphs, rects = parser.parse(f"${inner}$", dpi=200, prop=prop)
            
            fig = plt.figure(figsize=(width/200, height/200))
            plt.axis('off')
            plt.text(0, descent/height, f"${inner}$", fontproperties=prop, color='black', verticalalignment='baseline')
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            
            img = ImageReader(buf)
            iw, ih = img.getSize()
            wp, hp = iw * 72 / 200, ih * 72 / 200
            res = (buf, wp, hp, descent * 72 / 200)
            _MATH_IMG_CACHE[math_text] = res
            return res
        except Exception as e:
            # Safe logging for Windows CP949 encoding issues
            safe_inner = (inner or "").encode("ascii", "backslashreplace").decode("ascii")
            print(f"MATH_RENDER_ERROR: {str(e).encode('ascii', 'backslashreplace').decode('ascii')} for: {safe_inner}")
            if 'fig' in locals(): plt.close(fig)
            return None

    def _tokenize_text(text):
        return re.findall(r"\S+\s*", text)

    def _build_inline_lines(text, size, max_w):
        segments = _split_segments(text)
        tokens = []
        for typ, content in segments:
            if typ == "text":
                for tok in _tokenize_text(content):
                    tokens.append(("text", tok, None))
            else:
                for seg_type, seg_content in _split_math_text_segments(content):
                    if seg_type == "text":
                        for tok in _tokenize_text(seg_content):
                            tokens.append(("text", tok, None))
                    else:
                        tokens.append(("math", seg_content, None))

        lines = []
        line = []
        line_w = 0
        
        svg_enabled = use_svg_render and (HAS_SVGLIB or HAS_CAIROSVG)
        for typ, content, _ in tokens:
            if typ == "text":
                w = c.stringWidth(content, font_name, size)
                token = (typ, content, w, None, 0, 0)
            else:
                res = _render_math_svg_cached(content, size) if svg_enabled else None
                if res:
                    svg_text, wp, hp, desc = res
                    token = (typ, content, wp, ("svg", svg_text), hp, desc)
                    w = wp
                else:
                    png_res = _render_math_img_cached(content, size)
                    if png_res:
                        buf, wp, hp, desc = png_res
                        token = (typ, content, wp, ("png", buf), hp, desc)
                        w = wp
                    else:
                        plain = latex_to_unicode_shared(content)
                        w = c.stringWidth(plain, font_name, size)
                        token = ("text", plain, w, None, 0, 0)

            if line and line_w + w > max_w:
                lines.append(line)
                line = []
                line_w = 0
            line.append(token)
            line_w += w

        if line:
            lines.append(line)
        return lines

    def draw_text(text, size=11, gap=5, indent=0, color=(0,0,0)):
        global y
        text = _clean(text)
        if not text: return
        max_w = A4[0] - 100 - indent
        paragraphs = text.split("\n")
        
        if render_math_as_plain:
            c.setFont(font_name, size)
            c.setFillColorRGB(*color)
            for para in paragraphs:
                if not para.strip():
                    y -= (size + gap)
                    continue
                for ln in simpleSplit(para, font_name, size, max_w):
                    if y < 60: c.showPage(); y = A4[1]-50; c.setFont(font_name, size)
                    c.drawString(50 + indent, y, ln)
                    y -= (size + gap)
            return

        c.setFont(font_name, size)
        c.setFillColorRGB(*color)
        for para in paragraphs:
            if not para.strip():
                y -= (size + gap)
                continue
            for line_tokens in _build_inline_lines(para, size, max_w):
                if y < 60: c.showPage(); y = A4[1]-50; c.setFont(font_name, size)
                
                max_h = size
                for t in line_tokens:
                    if t[0] == "math": max_h = max(max_h, t[4])
                
                x = 50 + indent
                for t in line_tokens:
                    if t[0] == "text":
                        c.drawString(x, y, t[1])
                    else:
                        _, _, w, payload, hp, desc = t
                        if payload:
                            kind, data = payload
                            img_y = y - desc
                            if kind == "png":
                                c.drawImage(ImageReader(data), x, img_y, width=w, height=hp, mask='auto')
                            elif kind == "svg":
                                if HAS_SVGLIB:
                                    drawing = svg2rlg(BytesIO(data.encode("utf-8")))
                                    if drawing:
                                        dw = drawing.width or w
                                        dh = drawing.height or hp
                                        sx = (w / dw) if dw else 1
                                        sy = (hp / dh) if dh else 1
                                        drawing.scale(sx, sy)
                                        renderPDF.draw(drawing, c, x, img_y)
                                elif HAS_CAIROSVG:
                                    png_bytes = cairosvg.svg2png(bytestring=data.encode("utf-8"))
                                    c.drawImage(ImageReader(BytesIO(png_bytes)), x, img_y, width=w, height=hp, mask='auto')
                    x += t[2]
                y -= (max_h + gap)

    # 4. Main Execution
    output_path = data.get("pdf_path", "solution.pdf")
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    c = canvas.Canvas(output_path, pagesize=A4)
    y = A4[1] - 50

    if render_math_as_plain:
        for entry in data.get("entries", []):
            if entry.get("original"): entry["original"] = latex_to_unicode_shared(entry["original"])
            if entry.get("explanation"): entry["explanation"] = latex_to_unicode_shared(entry["explanation"])
            if entry.get("answer"): entry["answer"] = latex_to_unicode_shared(entry["answer"])

    c.setFont(font_name, 20); c.setFillColorRGB(0.1, 0.1, 0.4)
    c.drawCentredString(A4[0]/2, y, "해 설 지"); y -= 40
    
    for entry in data.get("entries", []):
        if y < 150: c.showPage(); y = A4[1]-50
        
        c.setFont(font_name, 13); c.setFillColorRGB(0.2, 0.3, 0.5)
        prob_label = f"문제 {entry.get('number') or ''}"
        title_text = str(entry.get("title") or "").strip()
        original_text = str(entry.get("original") or "").strip()
        show_title = True
        if include_original and original_text:
            first_line = original_text.splitlines()[0].strip() if original_text else ""
            if title_text and title_text in first_line:
                show_title = False
        if show_title:
            c.drawString(50, y, f"{prob_label} {title_text}")
        else:
            c.drawString(50, y, f"{prob_label}")
        y -= 25
        
        if include_original and entry.get("original"):
            draw_text(entry["original"], size=10, gap=4, indent=15, color=(0.3, 0.3, 0.3))
            y -= 10
        
        ans_text = str(entry.get('answer') or '').strip()
        if ans_text and ans_text not in {"-", "—"}:
            c.setFont(font_name, 11)
            c.setFillColorRGB(0.8, 0, 0)
            c.drawString(50, y, "정답: ")
            draw_text(ans_text, size=11, gap=8, indent=35, color=(0.8, 0, 0))
            y -= 5
        
        c.setFont(font_name, 11)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(50, y, "해설: ")
        # Ensure explanation starts on the next line to avoid overlap
        y -= (11 + 2)
        expl_content = entry.get("explanation", "").strip()
        expl_content = re.sub(r'^(정답|답)\s*:\s*.*?\n', '', expl_content, flags=re.IGNORECASE)
        draw_text(expl_content, size=11, gap=5, indent=15, color=(0, 0, 0))
        
        y -= 20
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(50, y+10, A4[0]-50, y+10); y -= 20
    
    c.save()
    print(f"PDF_PATH: {output_path}")
    print(f"PDF_SIZE: {os.path.getsize(output_path)}")
    if emit_base64:
        with open(output_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        print(f"PDF_BASE64: {encoded}")

except Exception:
    traceback.print_exc()
    sys.exit(1)
"""

def _build_pdf_code(payload: Dict[str, Any]) -> str:
    """PDF 생성용 파이썬 코드 생성"""
    payload_json = json.dumps(payload)
    payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")
    
    # [Fix] 의존성 있는 모든 함수 소스 포함
    latex_source = (
        inspect.getsource(_convert_latex_block) + "\n\n" + 
        inspect.getsource(latex_to_unicode_shared)
    )
    
    code = template.replace("{{PAYLOAD}}", payload_b64)
    code = code.replace("{{LATEX_TO_UNICODE_SHARED}}", latex_source)
    return code

def _extract_pdf_meta(stdout_lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """stdout 출력에서 PDF 메타데이터 추출"""
    pdf_path, pdf_name, pdf_size, pdf_base64, pdf_font, pdf_font_path, pdf_font_loaded = None, None, None, None, None, None, None
    for line in stdout_lines:
        if line.startswith("PDF_PATH:"): pdf_path = line.split(":", 1)[1].strip()
        elif line.startswith("PDF_NAME:"): pdf_name = line.split(":", 1)[1].strip()
        elif line.startswith("PDF_SIZE:"):
            try: pdf_size = int(line.split(":", 1)[1].strip())
            except: pass
        elif line.startswith("PDF_BASE64:"): pdf_base64 = line.split(":", 1)[1].strip()
        elif line.startswith("PDF_FONT:"): pdf_font = line.split(":", 1)[1].strip()
        elif line.startswith("PDF_FONT_PATH:"): pdf_font_path = line.split(":", 1)[1].strip()
        elif line.startswith("PDF_FONT_LOADED:"): pdf_font_loaded = line.split(":", 1)[1].strip().lower() == "true"
    return pdf_path, pdf_name, pdf_size, pdf_base64, pdf_font, pdf_font_path, pdf_font_loaded

def _resolve_local_pdf_path(pdf_file_name: str) -> str:
    """로컬 PDF 저장 경로 결정"""
    output_dir = os.path.join(os.getcwd(), "outputs", "solution")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, pdf_file_name)

def _merge_local_font_env(envs: Optional[dict], pdf_path: str) -> dict:
    return dict(envs or {})

def _execute_pdf_locally(code: str, envs: dict) -> Tuple[bool, List[str], List[str], Optional[Exception]]:
    """로컬에서 PDF 생성 코드 실행"""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        
        # [구조 개선] 바이너리 모드로 실행하여 인코딩 오류 원천 차단 및 UTF-8 강제
        result = subprocess.run(
            [sys.executable, temp_file],
            env={**os.environ, **envs, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=False
        )
        
        # 부모 프로세스에서 안전하게 디코딩 수행
        stdout_lines = result.stdout.decode("utf-8", errors="ignore").splitlines()
        stderr_lines = result.stderr.decode("utf-8", errors="ignore").splitlines()
        
        ok = result.returncode == 0
        return ok, stdout_lines, stderr_lines, None
    except Exception as e:
        return False, [], [], e
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
