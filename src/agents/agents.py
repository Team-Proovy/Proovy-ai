# 에이전트 레지스트리와 로더를 정의하는 모듈
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.workflows.maingraph import graph_builder as tutor_graph_builder
from schema import AgentInfo

# 기본 에이전트 키
DEFAULT_AGENT = "tutor"

# LangGraph의 다양한 에이전트 패턴을 다루기 위한 타입 별칭
# - @entrypoint 함수는 Pregel 을 반환
# - StateGraph().compile() 은 CompiledStateGraph 를 반환
AgentGraph = CompiledStateGraph | Pregel  # get_agent() 가 항상 반환하는 타입
AgentGraphLike = CompiledStateGraph | Pregel | None  # 레지스트리에 저장될 수 있는 타입

# 전역 checkpointer 저장 (lifespan에서 주입됨)
_checkpointer: Any = None


def set_checkpointer(checkpointer: Any) -> None:
    """lifespan에서 호출하여 checkpointer를 주입합니다.

    checkpointer가 None인 경우에도 정상적으로 설정되며,
    이 경우 그래프는 checkpointer 없이 컴파일됩니다.
    """
    global _checkpointer
    _checkpointer = checkpointer


@dataclass
class Agent:
    description: str
    graph_like: AgentGraphLike = None  # 지연 컴파일을 위해 None 허용
    graph_builder: Any = None  # 컴파일 전 builder


# 에이전트 레지스트리 - (key: 에이전트 ID, value: Agent)
agents: dict[str, Agent] = {
    "tutor": Agent(
        description="Proovy 수학/과학 튜터 에이전트",
        graph_like=None,
        graph_builder=tutor_graph_builder,
    ),
}


def get_agent(agent_id: str) -> AgentGraph:
    """필요하다면 지연 컴파일을 수행한 뒤 에이전트 그래프를 반환한다.

    _checkpointer가 None인 경우에도 그래프는 정상적으로 컴파일되며,
    단지 대화 히스토리가 저장되지 않을 뿐입니다.
    """
    agent = agents[agent_id]

    # 아직 컴파일되지 않았으면 checkpointer와 함께 컴파일
    # checkpointer가 None이어도 그래프는 정상 동작함 (히스토리 저장만 안됨)
    if agent.graph_like is None and agent.graph_builder is not None:
        agent.graph_like = agent.graph_builder.compile(checkpointer=_checkpointer)

    return agent.graph_like


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description)
        for agent_id, agent in agents.items()
    ]
