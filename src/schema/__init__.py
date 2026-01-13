"""Schema package public interface.

Re-export model enums and HTTP schema models so that

    from schema import AgentInfo, ChatMessage, ...

style imports work across the project.
"""

from .models import (
    AllModelEnum,
    OpenAIModelName,
    OpenRouterModelName,
    Provider,
)

from .schema import (
    AgentInfo,
    ServiceMetadata,
    UserInput,
    StreamInput,
    ToolCall,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ChatHistoryInput,
    ChatHistory,
)

__all__ = [
    "AllModelEnum",
    "OpenAIModelName",
    "OpenRouterModelName",
    "Provider",
    "AgentInfo",
    "ServiceMetadata",
    "UserInput",
    "StreamInput",
    "ToolCall",
    "ChatMessage",
    "Feedback",
    "FeedbackResponse",
    "ChatHistoryInput",
    "ChatHistory",
]
