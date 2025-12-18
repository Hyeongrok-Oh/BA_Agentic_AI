"""
Base classes for Agents and Tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import os
from openai import OpenAI


@dataclass
class ToolResult:
    """Tool 실행 결과"""
    success: bool
    data: Any = None
    error: str = None


@dataclass
class AgentContext:
    """Agent 실행 컨텍스트 (상태 관리)"""
    query: str
    history: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_step(self, step_name: str, result: Any):
        """실행 단계 기록"""
        self.history.append({
            "step": step_name,
            "result": result
        })


class BaseTool(ABC):
    """
    Tool 베이스 클래스

    특징:
    - 단일 기능 수행
    - 입력 → 출력 명확
    - 결정권 없음 (Agent가 호출)
    """

    name: str = "base_tool"
    description: str = "Base tool"

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Tool 실행"""
        pass


class BaseAgent(ABC):
    """
    Agent 베이스 클래스

    특징:
    - 자율성 (Autonomy)
    - Planning & Selection
    - 상태 관리 (Context)
    - 협업 가능
    """

    name: str = "base_agent"
    description: str = "Base agent"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(api_key=self.api_key)
        self.tools: List[BaseTool] = []
        self.sub_agents: List['BaseAgent'] = []

    def add_tool(self, tool: BaseTool):
        """Tool 추가"""
        self.tools.append(tool)

    def add_sub_agent(self, agent: 'BaseAgent'):
        """하위 Agent 추가"""
        self.sub_agents.append(agent)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """이름으로 Tool 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def get_sub_agent(self, name: str) -> Optional['BaseAgent']:
        """이름으로 하위 Agent 찾기"""
        for agent in self.sub_agents:
            if agent.name == name:
                return agent
        return None

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """LLM 호출 (Agent의 두뇌)

        o1/o3 추론 모델 지원:
        - o1, o1-mini, o1-preview, o3-mini 등
        - system prompt를 user message에 포함
        - temperature 파라미터 미사용
        """
        # o1/o3 추론 모델 여부 확인
        is_reasoning_model = model.startswith("o1") or model.startswith("o3")

        messages = []

        if is_reasoning_model:
            # o1/o3 모델: system prompt를 user message에 포함
            combined_prompt = prompt
            if system_prompt:
                combined_prompt = f"[시스템 지침]\n{system_prompt}\n\n[사용자 요청]\n{prompt}"
            messages.append({"role": "user", "content": combined_prompt})

            # o1/o3 모델용 API 호출 (temperature 없음, max_completion_tokens 사용)
            response = self.llm.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens
            )
        else:
            # 일반 모델 (gpt-4o, gpt-4o-mini 등)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

        return response.choices[0].message.content.strip()

    @abstractmethod
    def run(self, context: AgentContext) -> Dict[str, Any]:
        """
        Agent 실행

        Args:
            context: 실행 컨텍스트 (쿼리, 히스토리, 메타데이터)

        Returns:
            실행 결과 딕셔너리
        """
        pass
