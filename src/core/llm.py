# 설정된 모델 enum 을 실제 LangChain LLM(채팅 모델) 인스턴스로 변환하는 팩토리 모듈입니다.
from functools import cache
from typing import TypeAlias

from langchain_openai import ChatOpenAI

from core.settings import settings
from schema.models import AllModelEnum, OpenAIModelName, OpenRouterModelName

_MODEL_TABLE = {m: m.value for m in OpenAIModelName} | {
    m: m.value for m in OpenRouterModelName
}


ModelT: TypeAlias = ChatOpenAI


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # 참고: streaming=True 로 설정된 모델은 /stream 엔드포인트가
    # 기본값인 stream_tokens=True 로 호출되면 토큰을 생성되는 대로 스트리밍한다.
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        # OPENAI_API_KEY는 Settings에서 SecretStr로 관리되므로 여기서 직접 전달해준다.
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set to use OpenAI models")
        return ChatOpenAI(
            model=api_model_name,
            streaming=True,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
        )
    if model_name in OpenRouterModelName:
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY must be set to use OpenRouter models")
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )

    raise ValueError(f"Unsupported model: {model_name}")
