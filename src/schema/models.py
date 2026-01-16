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
    GEMINI_25_PRO = "google/gemini-2.5-pro"
    GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"
    GEMINI_3_PRO_PREVIEW = "google/gemini-3-pro-preview"

    CLAUDE_HAIKU_45 = "anthropic/claude-haiku-4.5"
    CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4"
    CLAUDE_SONNET_45 = "anthropic/claude-sonnet-4.5"
    CLAUDE_OPUS_45 = "anthropic/claude-opus-4.5"

    GPT_5 = "openai/gpt-5"
    GPT_5_CODEX = "openai/gpt-5-codex"
    GPT_5_1_OPENROUTER = "openai/gpt-5.1"
    GPT_5_1_CODEX = "openai/gpt-5.1-codex"
    GPT_5_1_CODEX_MAX = "openai/gpt-5.1-codex-max"
    GPT_5_1_CODEX_MINI = "openai/gpt-5.1-codex-mini"
    GPT_5_2 = "openai/gpt-5.2"
    GPT_5_2_CODEX = "openai/gpt-5.2-codex"


AllModelEnum: TypeAlias = OpenAIModelName | OpenRouterModelName
