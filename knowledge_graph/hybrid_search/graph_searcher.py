"""
Graph 검색기 - 검증된 가설과 관련된 Event/Factor를 Neo4j에서 검색
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase

from .hypothesis_generator import Hypothesis


@dataclass
class GraphEvidence:
    """그래프 검색 결과"""
    event_name: str
    event_category: str
    event_severity: str
    factor_name: str
    impact_type: str  # INCREASES, DECREASES
    magnitude: str
    evidence: str
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

    # 매출
    "매출": ["매출", "Revenue", "수익성"],
    "판매량": ["판매량", "출하량", "수요", "TV수요부진"],

    # 외부
    "환율": ["환율"],
    "경쟁": ["경쟁심화"],
}


class GraphSearcher:
    """Neo4j 그래프 검색기"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = "neo4j"
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def search_for_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        region: str = None
    ) -> Dict[str, List[GraphEvidence]]:
        """검증된 가설들에 대한 그래프 증거 검색"""

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
    ) -> List[GraphEvidence]:
        """단일 가설에 대한 검색"""

        # Factor 이름 매핑
        factor_names = self._get_factor_names(hypothesis.factor)

        # 가설 방향에 따른 관계 유형
        if hypothesis.direction.lower() == "increase":
            rel_type = "INCREASES"
        else:
            rel_type = "DECREASES"

        # Cypher 쿼리 실행
        evidences = []

        for factor_name in factor_names:
            query_results = self._execute_search(factor_name, rel_type, region)
            evidences.extend(query_results)

        return evidences

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
    ) -> List[GraphEvidence]:
        """Cypher 쿼리 실행"""

        # 지역 필터 조건 (올바른 Cypher 문법 사용)
        region_condition = ""
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

        # 지역 필터가 있으면 TARGETS 관계로 필터링
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

        evidences = []

        with self.driver.session(database=self.database) as session:
            try:
                params = {
                    "factor_name": factor_name,
                    "factor_id": factor_name.lower().replace(" ", "_")
                }
                if region_param:
                    params["region"] = region_param

                result = session.run(query, **params)

                for record in result:
                    evidences.append(GraphEvidence(
                        event_name=record["event_name"] or "",
                        event_category=record["event_category"] or "",
                        event_severity=record["event_severity"] or "medium",
                        factor_name=record["factor_name"] or factor_name,
                        impact_type=record["impact_type"] or rel_type,
                        magnitude=record["magnitude"] or "medium",
                        evidence=record["evidence"] or "",
                        target_regions=record["target_regions"] or []
                    ))

            except Exception as e:
                print(f"그래프 검색 오류: {e}")

        return evidences

    def search_causal_path(
        self,
        factor_name: str,
        anchor_name: str = "원가"
    ) -> List[Dict]:
        """Factor → Anchor 인과 경로 검색"""

        query = """
        MATCH path = (f:Factor)-[:AFFECTS|INFLUENCES*1..3]->(a:Anchor)
        WHERE f.name CONTAINS $factor_name AND a.name CONTAINS $anchor_name
        RETURN
            [n in nodes(path) | n.name] as path_nodes,
            [r in relationships(path) | type(r)] as path_relations,
            length(path) as path_length
        ORDER BY path_length
        LIMIT 5
        """

        paths = []

        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(
                    query,
                    factor_name=factor_name,
                    anchor_name=anchor_name
                )

                for record in result:
                    paths.append({
                        "nodes": record["path_nodes"],
                        "relations": record["path_relations"],
                        "length": record["path_length"]
                    })

            except Exception as e:
                print(f"경로 검색 오류: {e}")

        return paths

    def search_events_by_region(self, region: str, limit: int = 10) -> List[Dict]:
        """특정 지역에 영향을 주는 이벤트 검색"""

        query = """
        MATCH (e:Event)-[:TARGETS]->(d:Dimension {name: $region})
        OPTIONAL MATCH (e)-[r:INCREASES|DECREASES]->(f:Factor)
        RETURN
            e.name as event_name,
            e.category as category,
            e.severity as severity,
            collect(DISTINCT {factor: f.name, impact: type(r)}) as factor_impacts
        ORDER BY e.severity DESC
        LIMIT $limit
        """

        events = []

        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, region=region, limit=limit)

                for record in result:
                    events.append({
                        "name": record["event_name"],
                        "category": record["category"],
                        "severity": record["severity"],
                        "factor_impacts": record["factor_impacts"]
                    })

            except Exception as e:
                print(f"이벤트 검색 오류: {e}")

        return events

    def vector_search(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Vector 유사도 검색 (Neo4j Vector Index 사용)"""

        # 먼저 쿼리 텍스트를 임베딩으로 변환해야 함
        # 여기서는 간단히 키워드 기반 검색으로 대체

        query = """
        MATCH (e:Event)
        WHERE e.name CONTAINS $keyword OR e.evidence CONTAINS $keyword
        OPTIONAL MATCH (e)-[r:INCREASES|DECREASES]->(f:Factor)
        RETURN
            e.name as event_name,
            e.category as category,
            e.evidence as evidence,
            collect(DISTINCT f.name) as related_factors
        LIMIT $limit
        """

        results = []

        # 쿼리에서 키워드 추출 (간단한 방식)
        keywords = query_text.split()[:3]

        with self.driver.session(database=self.database) as session:
            for keyword in keywords:
                if len(keyword) < 2:
                    continue

                try:
                    result = session.run(query, keyword=keyword, limit=top_k)

                    for record in result:
                        results.append({
                            "event_name": record["event_name"],
                            "category": record["category"],
                            "evidence": record["evidence"],
                            "related_factors": record["related_factors"],
                            "matched_keyword": keyword
                        })

                except Exception as e:
                    print(f"벡터 검색 오류: {e}")

        # 중복 제거
        seen = set()
        unique_results = []
        for r in results:
            if r["event_name"] not in seen:
                seen.add(r["event_name"])
                unique_results.append(r)

        return unique_results[:top_k]
