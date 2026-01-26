# LangGraph 에이전트들이 공유해서 사용할 상태(state) 정의를 모아두는 곳입니다.
from typing import Annotated, Literal, List, Dict, Any, Optional

from typing_extensions import TypedDict, NotRequired
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


class ProblemAnalysis(BaseModel):
    problem_statement: str
    domain: str
    knowns: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    laws: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)


class SolveStrategy(BaseModel):
    summary: str
    steps: List[str] = Field(default_factory=list)
    generated_code: str


class ComputationSummary(BaseModel):
    success: bool
    stdout: List[str] = Field(default_factory=list)
    stderr: List[str] = Field(default_factory=list)
    text: Optional[str] = None


class ExplainResult(BaseModel):
    explanation: str = ""
    examples: List[str] = Field(default_factory=list)


class SolveResult(BaseModel):
    problem: Optional[str] = None
    analysis: Optional[ProblemAnalysis] = None
    strategy: Optional[SolveStrategy] = None
    computation: Optional[ComputationSummary] = None
    answer: Optional[str] = None
    steps: List[str] = Field(default_factory=list)
    latex: Optional[str] = None


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
    reasons: List[str] = Field(default_factory=list)



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    check_result: NotRequired[Literal["mixed_files", "image_only", "text_only"]]

    # Preprocessing Layer
    file_processing: FileProcessing

    # Router Layer
    router_state: RouterState

    # Feature Layer
    feature_results: Annotated[List[Dict[str, Any]], operator.add]
    solve_result: NotRequired[Optional[SolveResult]]
    explain_result: NotRequired[Optional[ExplainResult]]
    graph_result: NotRequired[Optional[GraphResult]]
    variant_result: NotRequired[Optional[VariantResult]]
    solution_result: NotRequired[Optional[SolutionResult]]

    # Review Layer
    review_state: NotRequired[ReviewState]

    # Routing context
    prev_action: Optional[str]
    next_action: Optional[str]
    simple_response: Optional[bool] = None
    retry_count: NotRequired[int]
    retry_limit_exceeded: Optional[bool] = None
    plan: NotRequired[List[str]]
    current_step: NotRequired[Optional[str]]

    # 업로드/툴/최종 (기본값 추가)
    input_files: NotRequired[List[str]]  # 업로드된 파일들의 로컬 경로 리스트
    tool_outputs: NotRequired[Dict[str, Any]]
    final_output: NotRequired[Dict[str, Any]]
    chosen_features: NotRequired[List[str]]
