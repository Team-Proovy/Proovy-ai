from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


_TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def _resolve_template_path(template_path: str) -> Path:
    root = _TEMPLATE_ROOT.resolve()
    candidate = (root / template_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"Invalid template path: {template_path}")
    return candidate


@lru_cache(maxsize=None)
def load_prompt(template_path: str) -> str:
    path = _resolve_template_path(template_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template_path: str, **kwargs: object) -> str:
    template = load_prompt(template_path)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in kwargs:
            raise KeyError(
                f"Missing template variable '{key}' for prompt: {template_path}"
            )
        return str(kwargs[key])

    return _PLACEHOLDER_PATTERN.sub(_replace, template)
