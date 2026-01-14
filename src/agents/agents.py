# 에이전트 레지스트리와 로더를 정의하는 모듈
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from schema import AgentInfo

# 기본 에이전트 키(아직 구현되지 않은 기본 에이전트)
DEFAULT_AGENT = ""

# LangGraph의 다양한 에이전트 패턴을 다루기 위한 타입 별칭
# - @entrypoint 함수는 Pregel 을 반환
# - StateGraph().compile() 은 CompiledStateGraph 를 반환
AgentGraph = CompiledStateGraph | Pregel  # get_agent() 가 항상 반환하는 타입
AgentGraphLike = CompiledStateGraph | Pregel  # 레지스트리에 저장될 수 있는 타입


@dataclass
class Agent:
    description: str
    graph_like: AgentGraphLike


# 에이전트 레지스트리 - (key: 에이전트 ID, value: Agent)
agents: dict[str, Agent] = {}


def get_agent(agent_id: str) -> AgentGraph:
    """필요하다면 지연 로딩을 수행한 뒤 에이전트 그래프를 반환한다."""
    agent_graph = agents[agent_id].graph_like
    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description)
        for agent_id, agent in agents.items()
    ]
