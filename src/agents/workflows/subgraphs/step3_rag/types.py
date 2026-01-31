from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class RetrievedDoc(TypedDict):
    id: str
    title: str
    text: str
    score: Optional[float]
    metadata: Dict[str, Any]


class RagOutput(TypedDict, total=False):
    retrieved_docs: List[RetrievedDoc]
    rag_relevance_passed: bool
    next_action: str


class RetrievedDocModel(BaseModel):
    id: str
    title: str
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

