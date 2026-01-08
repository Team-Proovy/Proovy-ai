# LLM 제공자와 모델 이름들을 타입으로 정리하는 모듈이다.

from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    OPENROUTER = auto()


class OpenAIModelName(StrEnum):
    """OpenAI GPT 계열 모델 이름 (https://platform.openai.com/docs/models/gpt-4o)."""

    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_1 = "gpt-5.1"


class OpenRouterModelName(StrEnum):
    """OpenRouter 에서 제공하는 모델 이름 (https://openrouter.ai/models)."""

    GEMINI_25_FLASH = "google/gemini-2.5-flash"


AllModelEnum: TypeAlias = OpenAIModelName | OpenRouterModelName
