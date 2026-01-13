"""Core package public interface.

Expose settings and model factory at the package level so that

    from core import settings, get_model

is supported.
"""

from .settings import settings
from .llm import get_model

__all__ = [
    "settings",
    "get_model",
]
