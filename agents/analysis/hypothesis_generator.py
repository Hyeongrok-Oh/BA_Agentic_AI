"""
Hypothesis Generator Agent - Graph-Based 가설 생성 에이전트

역할:
- Knowledge Graph에서 KPI 관련 모든 Factor 조회
- Factor → Hypothesis 직접 변환 (LLM 없이)
- 체계적이고 일관된 가설 생성
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..base import BaseAgent, AgentContext
from ..tools import GraphExecutor


@dataclass
class EventDetail:
    """이벤트 상세 정보"""
    name: str
    category: str           # geopolitical, policy, market, etc.
    severity: str           # critical, high, medium, low
    impact_direction: str   # INCREASES, DECREASES
    evidence: str           # 이벤트 설명/근거
    target_regions: List[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    """가설 데이터 클래스"""
    id: str
    category: str           # revenue, cost, pricing, external
    factor: str             # 관련 Factor
    direction: str          # increase, decrease
    description: str        # 가설 설명 (상세)
    reasoning: str = ""     # 인과관계 설명
    sql_template: str = ""  # 검증용 SQL 힌트
    validated: bool = None
    validation_data: Dict = field(default_factory=dict)
    # Graph 정보
    graph_evidence: Dict = field(default_factory=dict)
    # 관련 이벤트 상세
    related_events: List[EventDetail] = field(default_factory=list)


# KPI 매핑
KPI_MAPPING = {
    "매출": {"anchor_id": "revenue", "keywords": ["매출", "revenue", "수익", "sales"]},
    "원가": {"anchor_id": "cost", "keywords": ["원가", "cost", "비용", "expense"]},
    "판매수량": {"anchor_id": "quantity", "keywords": ["판매량", "수량", "quantity", "volume"]},
}

# Factor → Category 매핑 (Factor 이름 기반 카테고리 분류)
FACTOR_CATEGORY_KEYWORDS = {
    "cost": ["원가", "비용", "재료비", "물류비", "운임", "관세", "인건비", "오버헤드", "부품"],
    "revenue": ["매출", "판매", "수익", "수요", "시장", "점유율", "출하"],
    "pricing": ["가격", "할인", "프로모션", "PP", "MDF", "ASP"],
    "external": ["환율", "금리", "경기", "경쟁", "규제", "정책", "관세율"],
}

# KPI별 변동 설명 템플릿
KPI_CHANGE_TEMPLATES = {
    "매출": {
        "PROPORTIONAL": {"increase": "증가로 매출 상승", "decrease": "감소로 매출 하락"},
        "INVERSE": {"increase": "증가로 매출 하락", "decrease": "감소로 매출 상승"},
    },
    "원가": {
        "PROPORTIONAL": {"increase": "증가로 원가 상승", "decrease": "감소로 원가 하락"},
        "INVERSE": {"increase": "증가로 원가 하락", "decrease": "감소로 원가 상승"},
    },
    "판매수량": {
        "PROPORTIONAL": {"increase": "증가로 판매량 증가", "decrease": "감소로 판매량 감소"},
        "INVERSE": {"increase": "증가로 판매량 감소", "decrease": "감소로 판매량 증가"},
    },
}


class HypothesisGenerator(BaseAgent):
    """Graph-Enhanced 가설 생성 에이전트"""

    name = "hypothesis_generator"
    description = "Knowledge Graph를 활용하여 KPI 변동에 대한 가설을 생성합니다."

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.graph_executor = GraphExecutor()

    def generate(
        self,
        question: str,
        company: str = "LGE",
        period: str = None,
        region: str = None
    ) -> List[Hypothesis]:
        """
        Graph-Based 가설 생성 (Graph에서 Factor만 조회하여 가설 생성)

        Args:
            question: 분석 질문
            company: 회사 코드
            period: 분석 기간 (예: "2024년 Q4")
            region: 분석 지역 (예: "NA", "EU")

        Returns:
            가설 목록
        """
        # 1. 질문에서 KPI 추출
        target_kpi = self._extract_kpi(question)
        print(f"[HypothesisGenerator] 대상 KPI: {target_kpi}")

        # 2. Graph에서 해당 KPI(Anchor)와 연결된 모든 Factor 조회
        graph_factors = self._get_all_factors_for_anchor(target_kpi)
        print(f"[HypothesisGenerator] Graph Factor 수: {len(graph_factors)}개")

        # 3. 각 Factor를 직접 Hypothesis로 변환 (Event 조회 없이 단순 변환)
        hypotheses = []
        for i, factor_data in enumerate(graph_factors):
            hypothesis = self._convert_factor_to_hypothesis(
                index=i + 1,
                factor_data=factor_data,
                target_kpi=target_kpi
            )
            if hypothesis:
                hypotheses.append(hypothesis)

        print(f"[HypothesisGenerator] 생성된 가설 수: {len(hypotheses)}개")
        return hypotheses

    def _get_all_factors_for_anchor(self, kpi: str) -> List[Dict]:
        """Graph에서 KPI(Anchor)와 연결된 모든 Factor 조회"""
        anchor_id = KPI_MAPPING.get(kpi, {}).get("anchor_id", "cost")

        # 단순화된 쿼리 - Factor만 조회
        query = """
        MATCH (f:Factor)-[r:AFFECTS]->(a:Anchor {id: $anchor_id})
        RETURN
            f.name as factor_name,
            f.id as factor_id,
            f.category as factor_category,
            f.description as factor_description,
            r.type as relation_type,
            r.mention_count as mention_count,
            r.evidence as relation_evidence,
            a.name as anchor_name
        ORDER BY r.mention_count DESC
        """

        try:
            result = self.graph_executor.execute(query, {"anchor_id": anchor_id})
            if result.success and result.data:
                return result.data
        except Exception as e:
            print(f"[HypothesisGenerator] Factor 조회 오류: {e}")

        return []

    def _convert_factor_to_hypothesis(
        self,
        index: int,
        factor_data: Dict,
        target_kpi: str
    ) -> Optional[Hypothesis]:
        """Factor 데이터를 Hypothesis 객체로 변환 (단순 변환, Event 없음)"""
        factor_name = factor_data.get("factor_name", "")
        factor_id = factor_data.get("factor_id", "")
        factor_description = factor_data.get("factor_description", "") or ""
        relation_type = factor_data.get("relation_type", "PROPORTIONAL")
        relation_evidence = factor_data.get("relation_evidence", "") or ""
        mention_count = factor_data.get("mention_count", 0) or 0

        if not factor_name:
            return None

        # 카테고리 결정
        category = self._determine_category(factor_name)

        # 방향 결정: PROPORTIONAL이면 KPI와 같은 방향
        direction = "increase" if relation_type == "PROPORTIONAL" else "decrease"

        # 간단한 설명 생성
        relation_kr = "동비례" if relation_type == "PROPORTIONAL" else "역비례"
        description = f"{factor_name}이(가) {target_kpi}에 영향 ({relation_kr})"

        # Graph Evidence 구성
        graph_evidence = {
            "from_graph": True,
            "factor_id": factor_id,
            "factor_description": factor_description,
            "relation_type": relation_type,
            "relation_evidence": relation_evidence,
            "mention_count": mention_count
        }

        return Hypothesis(
            id=f"H{index}",
            category=category,
            factor=factor_name,
            direction=direction,
            description=description,
            reasoning="",  # 답변 생성 시 채워짐
            sql_template=self._generate_sql_hint(factor_name, category),
            graph_evidence=graph_evidence,
            related_events=[]  # 답변 생성 시 채워짐
        )

    def _determine_category(self, factor_name: str) -> str:
        """Factor 이름 기반 카테고리 결정"""
        factor_lower = factor_name.lower()

        for category, keywords in FACTOR_CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in factor_lower:
                    return category

        return "external"  # 기본값

    def _generate_sql_hint(self, factor_name: str, category: str) -> str:
        """SQL 검증 힌트 생성"""
        hints = {
            "cost": {
                "물류비": "COST_TYPE = 'LOG'",
                "재료비": "COST_TYPE = 'MAT'",
                "관세": "COST_TYPE = 'TAR'",
                "오버헤드": "COST_TYPE = 'OH'",
            },
            "pricing": {
                "할인": "COND_TYPE = 'K007'",
                "프로모션": "COND_TYPE = 'ZPRO'",
                "MDF": "COND_TYPE = 'ZMDF'",
            },
            "revenue": {
                "판매량": "SUM(QUANTITY)",
                "매출": "SUM(NET_VALUE)",
            }
        }

        category_hints = hints.get(category, {})
        for key, hint in category_hints.items():
            if key in factor_name:
                return hint

        return f"{factor_name} 기간 비교"

    def _extract_kpi(self, question: str) -> str:
        """질문에서 KPI 추출"""
        question_lower = question.lower()

        for kpi_name, info in KPI_MAPPING.items():
            for keyword in info["keywords"]:
                if keyword in question_lower:
                    return kpi_name

        # 기본값
        return "원가"

    def run(self, context: AgentContext) -> Dict[str, Any]:
        """Agent 실행"""
        question = context.query
        metadata = context.metadata or {}

        hypotheses = self.generate(
            question=question,
            company=metadata.get("company", "LGE"),
            period=metadata.get("period"),
            region=metadata.get("region")
        )

        result = {
            "hypotheses": [
                {
                    "id": h.id,
                    "category": h.category,
                    "factor": h.factor,
                    "direction": h.direction,
                    "description": h.description,
                    "reasoning": h.reasoning,
                    "graph_evidence": h.graph_evidence,
                    "related_events": [
                        {
                            "name": e.name,
                            "category": e.category,
                            "severity": e.severity,
                            "impact_direction": e.impact_direction,
                            "evidence": e.evidence,
                            "target_regions": e.target_regions
                        }
                        for e in h.related_events
                    ]
                }
                for h in hypotheses
            ],
            "count": len(hypotheses)
        }

        context.add_step("hypothesis_generation", result)
        return {"hypotheses": hypotheses, **result}
