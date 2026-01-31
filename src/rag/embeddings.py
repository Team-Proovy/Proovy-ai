from __future__ import annotations

from typing import List


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Mock embeddings generator — replace with real embedding call later."""
    return [[float(len(text) % 10), 0.0, 1.0] for text in texts]

