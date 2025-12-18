"""
Evidence Collector Agent - 증거 수집 에이전트

역할:
- Search Agent를 사용하여 검증된 가설과 관련된 외부 이벤트/요인 검색
- Knowledge Graph에서 인과관계 증거 수집
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from ..base import BaseAgent, AgentContext
from ..search_agent import SearchAgent
from .hypothesis_generator import Hypothesis


@dataclass
class Evidence:
    """증거 데이터 클래스"""
    event_name: str
    event_category: str
    event_severity: str
    factor_name: str
    impact_type: str  # INCREASES, DECREASES
    magnitude: str
    evidence_text: str
    target_regions: List[str]


# Factor 이름 매핑 (가설 Factor → Neo4j Factor 이름)
FACTOR_NAME_MAPPING = {
    # 원가
    "물류비": ["물류비", "해상운임", "운송비", "Logistics Cost"],
    "재료비": ["원재료비", "패널가격", "Material Cost"],
    "관세": ["관세", "Tariff"],
    # 가격
    "Price Protection": ["프로모션", "할인", "Price Protection", "마케팅비용"],
    "할인": ["할인", "Discount", "프로모션"],
    "MDF": ["MDF", "마케팅비용"],
    # 매출/수익
    "매출": ["매출", "Revenue", "수익성", "WebOS매출확대"],
    "수익": ["수익성", "원가개선"],
    # 판매량/수요 - 구체적인 Factor 이름들 추가
    "판매량": ["TV수요부진", "수요부진", "글로벌수요", "가전수요", "수요", "판매", "출하량", "소비심리위축"],
    "수요": ["TV수요부진", "수요부진", "글로벌수요", "가전수요", "수요", "소비심리위축"],
    # 외부
    "환율": ["환율"],
    "경쟁": ["경쟁심화", "프리미엄제품확대"],
    "경쟁사": ["경쟁심화"],
    # OLED
    "OLED": ["OLED판매확대", "OLED"],
    "프리미엄": ["프리미엄제품확대"],
}


class EvidenceCollector(BaseAgent):
    """증거 수집 에이전트"""

    name = "evidence_collector"
    description = "Knowledge Graph에서 검증된 가설과 관련된 외부 이벤트 증거를 수집합니다."

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.search_agent = SearchAgent(api_key)
        self.add_sub_agent(self.search_agent)

    def collect(
        self,
        hypotheses: List[Hypothesis],
        region: str = None
    ) -> Dict[str, List[Evidence]]:
        """
        검증된 가설들에 대한 증거 수집

        Args:
            hypotheses: 검증된 가설 목록
            region: 지역 필터

        Returns:
            {hypothesis_id: [Evidence, ...]}
        """
        results = {}

        for hypothesis in hypotheses:
            if not hypothesis.validated:
                continue

            evidences = self._search_single(hypothesis, region)
            if evidences:
                results[hypothesis.id] = evidences

        return results

    def _search_single(
        self,
        hypothesis: Hypothesis,
        region: str = None
    ) -> List[Evidence]:
        """단일 가설에 대한 증거 검색"""

        # Factor 이름 매핑
        factor_names = self._get_factor_names(hypothesis.factor)

        # 가설 방향에 따른 관계 유형
        if hypothesis.direction.lower() == "increase":
            rel_type = "INCREASES"
        else:
            rel_type = "DECREASES"

        evidences = []

        for factor_name in factor_names:
            # 먼저 원래 관계 유형으로 검색
            query_results = self._execute_search(factor_name, rel_type, region)
            evidences.extend(query_results)

            # "부진", "위축" 같은 부정적 Factor는 반대 관계도 검색
            # 예: "판매량 감소" → "수요부진 INCREASES" 도 검색
            if self._is_negative_factor(factor_name):
                opposite_rel = "INCREASES" if rel_type == "DECREASES" else "DECREASES"
                query_results = self._execute_search(factor_name, opposite_rel, region)
                evidences.extend(query_results)

        return evidences

    def _is_negative_factor(self, factor_name: str) -> bool:
        """부정적 의미의 Factor인지 확인"""
        negative_keywords = ["부진", "위축", "감소", "하락", "약화", "악화"]
        return any(kw in factor_name for kw in negative_keywords)

    def _get_factor_names(self, factor: str) -> List[str]:
        """가설 Factor를 Neo4j Factor 이름으로 매핑"""
        for key, names in FACTOR_NAME_MAPPING.items():
            if key.lower() in factor.lower() or factor.lower() in key.lower():
                return names
        return [factor]

    def _execute_search(
        self,
        factor_name: str,
        rel_type: str,
        region: str = None
    ) -> List[Evidence]:
        """Cypher 쿼리 실행"""

        # 지역 파라미터 설정
        region_param = None
        if region:
            region_upper = region.upper()
            if region_upper in ["NA", "NORTH AMERICA", "북미"]:
                region_param = "NA"
            elif region_upper in ["EU", "EUROPE", "유럽"]:
                region_param = "EU"
            elif region_upper in ["KR", "KOREA", "한국"]:
                region_param = "KR"
            elif region_upper in ["ASIA", "아시아"]:
                region_param = "ASIA"

        # Cypher 쿼리 생성
        if region_param:
            query = f"""
            MATCH (e:Event)-[r:{rel_type}]->(f:Factor)
            WHERE (f.name CONTAINS $factor_name OR f.id CONTAINS $factor_id)
            OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
            WITH e, r, f, collect(DISTINCT d.name) as target_regions
            WHERE $region IN target_regions OR size(target_regions) = 0
            RETURN
                e.name as event_name,
                e.category as event_category,
                e.severity as event_severity,
                f.name as factor_name,
                type(r) as impact_type,
                r.magnitude as magnitude,
                e.evidence as evidence,
                target_regions
            ORDER BY e.severity DESC
            LIMIT 10
            """
            params = {
                "factor_name": factor_name,
                "factor_id": factor_name.lower().replace(" ", "_"),
                "region": region_param
            }
        else:
            query = f"""
            MATCH (e:Event)-[r:{rel_type}]->(f:Factor)
            WHERE f.name CONTAINS $factor_name OR f.id CONTAINS $factor_id
            OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
            RETURN
                e.name as event_name,
                e.category as event_category,
                e.severity as event_severity,
                f.name as factor_name,
                type(r) as impact_type,
                r.magnitude as magnitude,
                e.evidence as evidence,
                collect(DISTINCT d.name) as target_regions
            ORDER BY e.severity DESC
            LIMIT 10
            """
            params = {
                "factor_name": factor_name,
                "factor_id": factor_name.lower().replace(" ", "_")
            }

        evidences = []

        try:
            result = self.search_agent.graph_tool.execute(query, params)

            if result.success and result.data:
                for record in result.data:
                    evidences.append(Evidence(
                        event_name=record.get("event_name") or "",
                        event_category=record.get("event_category") or "",
                        event_severity=record.get("event_severity") or "medium",
                        factor_name=record.get("factor_name") or factor_name,
                        impact_type=record.get("impact_type") or rel_type,
                        magnitude=record.get("magnitude") or "medium",
                        evidence_text=record.get("evidence") or "",
                        target_regions=record.get("target_regions") or []
                    ))

        except Exception as e:
            print(f"증거 수집 오류: {e}")

        return evidences

    def run(self, context: AgentContext) -> Dict[str, Any]:
        """Agent 실행"""
        hypotheses = context.metadata.get("validated_hypotheses", [])
        region = context.metadata.get("region")

        evidences = self.collect(
            hypotheses=hypotheses,
            region=region
        )

        # Evidence를 직렬화 가능한 형태로 변환
        serialized = {}
        for h_id, ev_list in evidences.items():
            serialized[h_id] = [
                {
                    "event_name": ev.event_name,
                    "event_category": ev.event_category,
                    "impact_type": ev.impact_type,
                    "factor_name": ev.factor_name,
                    "evidence": ev.evidence_text[:200] if ev.evidence_text else ""
                }
                for ev in ev_list
            ]

        result = {
            "evidences": evidences,
            "evidences_serialized": serialized,
            "hypothesis_count": len(evidences)
        }

        context.add_step("evidence_collection", {
            "hypothesis_count": len(evidences),
            "total_evidences": sum(len(v) for v in evidences.values())
        })

        return result
