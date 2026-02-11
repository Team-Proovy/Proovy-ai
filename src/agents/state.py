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
    ocr_blocks: Optional[Dict[str, Any]] = None  # {"pages": [...], "captions": [...]}


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
    guide: str = ""
    chunk_index: int = 0
    chunk_size: int = 0
    total_problems: int = 0
    total_chunks: int = 0
    problems: List[str] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)
    chunk_summary: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_file_name: Optional[str] = None
    pdf_mime_type: Optional[str] = None
    pdf_file_size: Optional[int] = None
    pdf_error: Optional[str] = None


class SolutionProgress(BaseModel):
    chunk_size: int = 5
    current_chunk: int = 0
    total_problems: int = 0
    total_chunks: int = 0
    done: bool = False


class ReviewState(BaseModel):
    """Review Layer (루프 제어 강화)"""

    passed: bool
    feedback: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    retry_count: int = 0  # 무한 루프 방지
    reasons: List[str] = Field(default_factory=list)


class CreditState(BaseModel):
    """Credit Layer (크레딧 관리)

    미들웨어 기반 비용 추적:
    - balance: 시작 시 조회한 잔액
    - total_cost: 미들웨어가 누적할 비용 (Annotated[float, add] 패턴)
    - 각 노드 실행 후 Conditional Edge에서 잔액 체크
    """

    balance: float = 0.0  # 시작 시 조회한 잔액
    total_cost: float = 0.0  # 누적 비용 (미들웨어가 자동 합산)
    cost_per_node: Dict[str, float] = Field(default_factory=dict)  # 노드별 비용
    difficulty: Literal["easy", "medium", "hard"] = "easy"  # 문제 난이도
    insufficient: bool = False  # 크레딧 부족 여부
    stopped_at_feature: Optional[str] = None  # 크레딧 부족으로 중단된 기능


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

    # Credit Layer
    credit_state: NotRequired[CreditState]

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
    problems: NotRequired[List[Dict[str, Any]]]  # 문제 인벤토리: [{"number": int|None, "text": str, "marker": str}]
    current_problem_index: NotRequired[int]  # 현재 풀이 대상 문제 인덱스(0-based)
    last_input_hash: NotRequired[str]  # input_files 변경 감지용 fingerprint
    solution_chunks: NotRequired[List[str]]
    solution_progress: NotRequired[SolutionProgress]
    solution_pdf: NotRequired[Dict[str, Any]]

    # Agent Writer Layer
    partial_responses: NotRequired[
        List[Dict[str, Any]]
    ]  # Writer 노드가 생성한 부분 응답들
