from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

from langchain_openai import OpenAIEmbeddings

from schema.models import OpenAIEmbeddingModelName

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embedding_client() -> OpenAIEmbeddings:
    """Create a real embeddings client using OpenRouter first."""
    # Enum을 사용하여 기본값 설정
    default_model = f"openai/{OpenAIEmbeddingModelName.TEXT_EMBEDDING_3_SMALL.value}"
    model = os.getenv("RAG_EMBEDDING_MODEL", default_model)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        logger.info("Embeddings provider=openrouter model=%s", model)
        return OpenAIEmbeddings(
            model=model,
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
    if openai_key:
        # OpenAI 직접 연결 시에는 접두사 없이 Enum 값 사용
        openai_default = OpenAIEmbeddingModelName.TEXT_EMBEDDING_3_SMALL.value
        logger.info(
            "Embeddings provider=openai model=%s",
            os.getenv("RAG_EMBEDDING_MODEL_OPENAI", openai_default),
        )
        return OpenAIEmbeddings(
            model=os.getenv("RAG_EMBEDDING_MODEL_OPENAI", openai_default),
            api_key=openai_key,
        )
    raise ValueError(
        "Embedding API key is missing. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings via OpenRouter/OpenAI-compatible embeddings API."""
    if not texts:
        return []
    client = _get_embedding_client()
    cleaned = [(text or "").strip() for text in texts]
    if not any(cleaned):
        return []
    try:
        vectors = client.embed_documents(cleaned)
    except Exception:
        logger.exception(
            "Embedding request failed. "
            "If using OpenRouter with openai/gpt-5-mini, check provider support for embeddings."
        )
        raise
    return [[float(v) for v in row] for row in vectors]


def embed_query(text: str) -> List[float]:
    """Generate a single embedding for a search query."""
    if not text or not text.strip():
        return []
    client = _get_embedding_client()
    try:
        vector = client.embed_query(text.strip())
        return [float(v) for v in vector]
    except Exception:
        logger.exception("Query embedding failed.")
        raise

