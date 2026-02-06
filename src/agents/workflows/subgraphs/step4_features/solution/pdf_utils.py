import json
import os
import re
import base64
import subprocess
import tempfile
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

# --- [Single Source of Truth] LaTeX to Plain Text Conversion Logic ---
def latex_to_unicode_shared(text: str) -> str:
    """LaTeX을 사람이 읽기 좋은 유니코드/플레인 텍스트로 변환하는 최상급 렌더러"""
    if not text: return ""
    
    # 1. 기초 정규화: 델리미터 및 불필요한 명령어 제거
    t = text.replace('\\\\', '\\').strip()
    if t.startswith('$') and t.endswith('$'): t = t[1:-1]
    elif t.startswith(r'\(') and t.endswith(r'\)'): t = t[2:-2]
    elif t.startswith(r'\[') and t.endswith(r'\]'): t = t[2:-2]
    
    # 2. 구조적 변환 (AST-like handling for operators)
    t = re.sub(r'\\sum(?:_\{([^}]+)\})?(?:\^\{([^}]+)\})?', 
               lambda m: f"Σ({m.group(1) or ''}→{m.group(2) or ''})", t)
    t = re.sub(r'\\int(?:_\{([^}]+)\})?(?:\^\{([^}]+)\})?', 
               lambda m: f"∫({m.group(1) or ''}→{m.group(2) or ''})", t)
    t = re.sub(r'\\lim_\{([^}]+)\}', lambda m: f"lim({m.group(1).replace(r'\to', '→')})", t)
    t = re.sub(r'\\(?:d|t)?frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', t)
    t = re.sub(r'\\sqrt(?:\[([^\]]+)\])?\{([^}]+)\}', 
               lambda m: f"{m.group(1) or ''}√{m.group(2)}", t)
    
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
    for k, v in mapping.items():
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

# --- PDF Generation Template (E2B Sandbox) ---
template = r"""
import os
import re
import json
import base64
import sys
import traceback
from io import BytesIO

# 1. LaTeX to Unicode Shared Logic
def latex_to_unicode_shared(text):
    if not text: return ""
    t = text.replace('\\', '\x01').replace('\x01\x01', '\x01').replace('\x01', '\\').strip()
    if t.startswith('$') and t.endswith('$'): t = t[1:-1]
    elif t.startswith('\\(') and t.endswith('\\)'): t = t[2:-2]
    
    t = re.sub(r'\\sum(?:_\{([^}]+)\})?(?:\^\{([^}]+)\})?', lambda m: f"Σ({m.group(1) or ''}→{m.group(2) or ''})", t)
    t = re.sub(r'\\int(?:_\{([^}]+)\})?(?:\^\{([^}]+)\})?', lambda m: f"∫({m.group(1) or ''}→{m.group(2) or ''})", t)
    t = re.sub(r'\\lim_\{([^}]+)\}', lambda m: f"lim({m.group(1).replace('\\to', '→')})", t)
    t = re.sub(r'\\(?:d|t)?frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', t)
    t = re.sub(r'\\sqrt(?:\[([^\]]+)\])?\{([^}]+)\}', lambda m: f"{m.group(1) or ''}√{m.group(2)}", t)
    
    mapping = {
        '\alpha': 'α', '\beta': 'β', '\gamma': 'γ', '\delta': 'δ', '\epsilon': 'ε',
        '\infty': '∞', '\to': '→', '\times': '×', '\cdot': '·', '\neq': '≠',
        '\sin': 'sin', '\cos': 'cos', '\tan': 'tan', '\ln': 'ln', '\log': 'log',
    }
    for k, v in mapping.items(): t = t.replace(k, v)
    
    sup_map = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
    sub_map = str.maketrans("0123456789+-=()n", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ")
    t = re.sub(r'\^\{?([0-9+\-=()n]+)\}?', lambda m: m.group(1).translate(sup_map), t)
    t = re.sub(r'_\{?([0-9+\-=()n]+)\}?', lambda m: m.group(1).translate(sub_map), t)
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    return t.replace('{', '').replace('}', '').strip()

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
    data = json.loads(base64.b64decode("{{PAYLOAD}}").decode("utf-8"))
    use_math_render = os.getenv("SOLUTION_USE_MATH_RENDER", "0") == "1"
    render_math_as_plain = os.getenv("SOLUTION_RENDER_MATH_AS_PLAIN", "1") == "1"
    complexity_threshold = int(os.getenv("SOLUTION_MATH_COMPLEXITY_THRESHOLD", "3"))
    include_original = os.getenv("SOLUTION_INCLUDE_ORIGINAL", "1") == "1"

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

    print(f"PDF_RENDERER_STATUS: font={font_name} loaded={font_loaded}")
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    def _clean(t):
        if not t: return ""
        return re.sub(r'[¤□\xa0]', ' ', t).strip()

    def _split_segments(text):
        if not text: return []
        math_re = re.compile(r'(\$[^$]+\$|\\\(.*?\\\)|\\\[.*?\\\])')
        parts = []
        idx = 0
        for m in math_re.finditer(text):
            if m.start() > idx: parts.append(("text", text[idx:m.start()]))
            parts.append(("math", m.group(0)))
            idx = m.end()
        if idx < len(text): parts.append(("text", text[idx:]))
        return parts

    def _is_complex_latex(s):
        if not s: return False
        cnt_cmds = len(re.findall(r"\\[a-zA-Z]+", s))
        has_env = bool(re.search(r"\\begin\{|matrix|pmatrix|cases|\\displaystyle", s))
        has_frac = bool(re.search(r"\\(?:d|t)?frac\s*\{[^}]*\}\s*\{[^}]*\}", s))
        return (cnt_cmds >= complexity_threshold) or has_env or has_frac or (len(s) > 120)

    _MATH_IMG_CACHE = {}

    def _render_math_img_cached(math_text, size):
        if math_text in _MATH_IMG_CACHE: return _MATH_IMG_CACHE[math_text]
        inner = math_text.replace('\\\\', '\\').strip()
        if inner.startswith('$') and inner.endswith('$'): inner = inner[1:-1]
        inner = re.sub(r'([가-힣\s]+)', lambda m: f"\\text{{{m.group(0).strip()}}}", inner)
        
        fig = plt.figure(figsize=(max(0.1, len(inner)*0.15), max(0.1, size/15)))
        plt.axis('off')
        plt.text(0, 0.5, f"${inner}$", fontsize=size, color='black', verticalalignment='center')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        _MATH_IMG_CACHE[math_text] = buf
        return buf

    def draw_text(text, size=11, gap=5, indent=0, color=(0,0,0)):
        global y
        text = _clean(text)
        if not text: return
        max_w = A4[0] - 100 - indent
        
        if render_math_as_plain:
            c.setFont(font_name, size)
            c.setFillColorRGB(*color)
            for ln in simpleSplit(text, font_name, size, max_w):
                if y < 50: c.showPage(); y = A4[1]-50; c.setFont(font_name, size)
                c.drawString(50 + indent, y, ln)
                y -= (size + gap)
            return

        for typ, content in _split_segments(text):
            if typ == "text":
                c.setFont(font_name, size); c.setFillColorRGB(*color)
                for ln in simpleSplit(content, font_name, size, max_w):
                    if y < 50: c.showPage(); y = A4[1]-50; c.setFont(font_name, size)
                    c.drawString(50+indent, y, ln)
                    y -= (size+gap)
            else:
                if not render_math_as_plain and use_math_render and _is_complex_latex(content):
                    try:
                        buf = _render_math_img_cached(content, size)
                        if buf:
                            img = ImageReader(buf)
                            iw, ih = img.getSize()
                            wp, hp = iw * 72 / 200, ih * 72 / 200
                            if wp > max_w: hp *= (max_w/wp); wp = max_w
                            if y < hp + 50: c.showPage(); y = A4[1]-50
                            c.drawImage(img, 50 + indent, y - hp, width=wp, height=hp, mask='auto')
                            y -= (hp + gap)
                            continue
                    except:
                        pass # Fallback to plain
                
                plain = latex_to_unicode_shared(content)
                c.setFont(font_name, size); c.setFillColorRGB(*color)
                for ln in simpleSplit(plain, font_name, size, max_w):
                    if y < 50: c.showPage(); y = A4[1]-50; c.setFont(font_name, size)
                    c.drawString(50+indent, y, ln)
                    y -= (size+gap)

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

    c.setFont(font_name, 18); c.drawCentredString(A4[0]/2, y, "해 설 지"); y -= 40
    for entry in data.get("entries", []):
        if y < 150: c.showPage(); y = A4[1]-50
        c.setFont(font_name, 12); c.setFillColorRGB(0.2, 0.3, 0.5)
        prob_label = f"문제 {entry.get('number') or ''}"
        c.drawString(50, y, f"{prob_label} {entry.get('title', '')}")
        y -= 25
        if include_original and entry.get("original"):
            draw_text(entry["original"], size=10, gap=5, indent=10, color=(0.3, 0.3, 0.3))
            y -= 10
        draw_text(f"정답: {entry.get('answer', '-')}", size=11, gap=8, indent=0, color=(0.8, 0, 0))
        draw_text("해설:", size=11, gap=5, indent=0, color=(0, 0, 0))
        draw_text(entry.get("explanation", ""), size=11, gap=5, indent=5, color=(0, 0, 0))
        y -= 25
        c.line(50, y+10, A4[0]-50, y+10); y -= 20
    
    c.save()
    print(f"PDF_PATH: {output_path}")
    print(f"PDF_SIZE: {os.path.getsize(output_path)}")

except Exception:
    traceback.print_exc()
    sys.exit(1)
"""

def _build_pdf_code(payload: Dict[str, Any]) -> str:
    """PDF 생성용 파이썬 코드 생성"""
    payload_json = json.dumps(payload)
    payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")
    code = template.replace("{{PAYLOAD}}", payload_b64)
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
        
        # [구조 개선] 바이너리 모드로 실행하여 인코딩 오류 원천 차단
        result = subprocess.run(
            [sys.executable, temp_file],
            env={**os.environ, **envs},
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
