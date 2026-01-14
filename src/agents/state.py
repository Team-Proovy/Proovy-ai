# LangGraph 에이전트들이 공유해서 사용할 상태(state) 정의를 모아두는 곳입니다.
from typing import TypedDict, Annotated, Literal, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator
from pydantic import BaseModel, Field


class FileProcessing(BaseModel):
    """Preprocessing Layer"""

    file_type: Literal["pdf", "ppt", "image", "text", "canvas"]
    converted_images: Optional[List[str]] = Field(default_factory=list)
    ocr_text: Optional[Dict[str, Any]] = None  # {"pages": [...], "full_text": "..."}


class RouterState(TypedDict):
    """Router Layer"""

    intent: Literal["solve", "explain", "graph", "variant", "solution", "greeting"]
    difficulty: Literal["easy", "medium", "hard"]
    target_feature: Literal["solve", "explain", "graph", "variant", "solution"]


class SolveResult(BaseModel):
    answer: str
    steps: List[str]
    latex: Optional[str] = None


class ExplainResult(BaseModel):
    explanation: str
    examples: List[str]


class GraphResult(BaseModel):
    mermaid: str
    image_url: Optional[str] = None


class VariantResult(BaseModel):
    problems: List[str]


class SolutionResult(BaseModel):
    guide: str


class ReviewState(BaseModel):
    """Review Layer (루프 제어 강화)"""

    passed: bool
    feedback: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    retry_count: int = 0  # 무한 루프 방지


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    # Preprocessing Layer
    file_processing: FileProcessing

    # Router Layer
    router_state: RouterState

    # Feature Layer
    feature_results: Annotated[List[Dict[str, Any]], operator.add]
    solve_result: Optional[SolveResult] = None
    explain_result: Optional[ExplainResult] = None
    graph_result: Optional[GraphResult] = None
    variant_result: Optional[VariantResult] = None
    solution_result: Optional[SolutionResult] = None

    # Review Layer
    review_state: ReviewState

    # Tool/최종 (기본값 추가)
    tool_outputs: Dict[str, Any] = {}
    final_output: Dict[str, Any] = {}
