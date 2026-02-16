from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# OpenAI 클라이언트 (OpenRouter 호환, lazy initialization)
_openai_client: Optional["OpenAI"] = None


def _get_openai_client() -> "OpenAI":
    """OpenAI 호환 클라이언트를 lazy하게 초기화하여 반환.

    OpenRouter 엔드포인트를 사용하여 임베딩을 생성합니다.
    """
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from err

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for embeddings"
            )

        _openai_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openai_client


def embed_texts(
    texts: List[str],
    *,
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> List[List[float]]:
    """OpenRouter Embeddings API를 사용하여 텍스트를 벡터로 변환.

    Args:
        texts: 임베딩할 텍스트 리스트
        model: 임베딩 모델명 (기본값: EMBEDDING_MODEL 환경변수 또는 openai/text-embedding-3-small)
        batch_size: 배치 크기 (기본값: EMBEDDING_BATCH_SIZE 환경변수 또는 100)

    Returns:
        각 텍스트에 대한 임베딩 벡터 리스트

    Raises:
        ValueError: API 키가 없거나 텍스트가 비어있는 경우
        Exception: API 호출 실패 시
    """
    if not texts:
        return []

    # 환경변수에서 설정 로드
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    if batch_size is None:
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

    # 빈 문자열 필터링 및 전처리
    processed_texts = []
    original_indices = []
    for i, text in enumerate(texts):
        cleaned = (text or "").strip()
        if cleaned:
            processed_texts.append(cleaned)
            original_indices.append(i)

    if not processed_texts:
        logger.warning("All texts were empty after preprocessing")
        dim = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        return [[0.0] * dim for _ in range(len(texts))]

    try:
        client = _get_openai_client()
        all_embeddings: List[List[float]] = []

        # 배치 처리
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i : i + batch_size]
            logger.debug(
                "Embedding batch %d-%d of %d texts",
                i,
                min(i + batch_size, len(processed_texts)),
                len(processed_texts),
            )

            response = client.embeddings.create(
                input=batch,
                model=model,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        # 원본 인덱스에 맞게 결과 재배열 (빈 텍스트는 0 벡터)
        embedding_dim = len(all_embeddings[0]) if all_embeddings else int(
            os.getenv("EMBEDDING_DIMENSION", "1536")
        )
        result = [[0.0] * embedding_dim for _ in range(len(texts))]
        for idx, embedding in zip(original_indices, all_embeddings):
            result[idx] = embedding

        logger.info(
            "Successfully embedded %d texts (model=%s, dim=%d)",
            len(processed_texts),
            model,
            embedding_dim,
        )
        return result

    except Exception as e:
        logger.exception("Failed to embed texts: %s", str(e))
        raise


def embed_query(
    query: str,
    *,
    model: Optional[str] = None,
) -> List[float]:
    """단일 쿼리 텍스트를 벡터로 변환.

    Args:
        query: 임베딩할 쿼리 텍스트
        model: 임베딩 모델명

    Returns:
        쿼리에 대한 임베딩 벡터
    """
    result = embed_texts([query], model=model)
    return result[0] if result else []


# Mock 함수 (테스트용, 실제 API를 사용하지 않음)
def embed_texts_mock(texts: List[str]) -> List[List[float]]:
    """Mock embeddings generator for testing without API calls."""
    dim = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    return [[float((len(text) + i) % 10) / 10.0] * dim for i, text in enumerate(texts)]
