"""
Multi-Agent System for LG Electronics HE Business Intelligence

서비스:
- Data Q&A: 데이터 조회 및 원인 분석
- Report Generation: 종합 보고서 생성

구조:
agents/
├── base.py                 # BaseAgent, BaseTool
├── tools/                  # 도구들
│   ├── sql_executor.py    # SQL 실행
│   └── graph_executor.py  # Cypher 실행
├── search_agent.py         # 데이터 검색 Agent
├── analysis/               # 분석 Agent 그룹
│   ├── hypothesis_generator.py
│   ├── hypothesis_validator.py
│   ├── evidence_collector.py
│   └── analysis_agent.py
├── report/                 # 보고서 Agent
│   └── report_agent.py
└── orchestrator.py         # Intent 라우터
"""

from .base import BaseAgent, BaseTool, AgentContext, ToolResult
from .tools import SQLExecutor, GraphExecutor
from .search_agent import SearchAgent
from .analysis import (
    HypothesisGenerator,
    HypothesisValidator,
    EvidenceCollector,
    AnalysisAgent
)
from .report import ReportAgent
from .orchestrator import Orchestrator

__all__ = [
    # Base
    "BaseAgent",
    "BaseTool",
    "AgentContext",
    "ToolResult",
    # Tools
    "SQLExecutor",
    "GraphExecutor",
    # Agents
    "SearchAgent",
    "HypothesisGenerator",
    "HypothesisValidator",
    "EvidenceCollector",
    "AnalysisAgent",
    "ReportAgent",
    # Orchestrator
    "Orchestrator",
]


def create_orchestrator(api_key: str = None, db_path: str = None) -> Orchestrator:
    """Orchestrator 생성 헬퍼 함수"""
    return Orchestrator(api_key=api_key, db_path=db_path)


def quick_query(query: str, verbose: bool = True) -> dict:
    """
    빠른 질문 처리 함수

    사용 예:
        from agents import quick_query
        result = quick_query("2024년 4분기 북미 영업이익이 왜 감소했어?")
    """
    orchestrator = Orchestrator()
    return orchestrator.process_query(query, verbose=verbose)
