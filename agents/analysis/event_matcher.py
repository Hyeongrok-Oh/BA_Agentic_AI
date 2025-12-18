"""
Event Matcher Agent - 하이브리드 스코어링 (Vector + Graph)

역할:
- Vector Similarity: 가설과 Event 간 의미적 유사도
- Graph Score: Knowledge Graph 관계 강도 (magnitude, direction)
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..base import BaseAgent, AgentContext
from ..search_agent import SearchAgent
from .hypothesis_generator import Hypothesis


@dataclass
class MatchedEvent:
    """매칭된 이벤트"""
    event_id: str
    event_name: str
    event_category: str
    severity: str
    is_ongoing: bool
    # Factor 관계
    matched_factor: str
    impact_type: str  # INCREASES, DECREASES
    magnitude: str
    # 매칭 점수
    total_score: float
    score_breakdown: Dict[str, float]
    # 출처
    sources: List[Dict]
    evidence: str
    target_regions: List[str]


# Magnitude 점수 매핑
MAGNITUDE_SCORES = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3
}

# Severity 점수 매핑
SEVERITY_SCORES = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2
}


class EventMatcher(BaseAgent):
    """하이브리드 이벤트 매칭 에이전트 (Vector + Graph)"""

    name = "event_matcher"
    description = "Vector Similarity + Knowledge Graph 기반 하이브리드 스코어링으로 Event를 매칭합니다."

    # 스코어 가중치
    WEIGHTS = {
        "semantic": 0.4,      # Vector Similarity (40%)
        "graph": 0.6,         # Graph Score (60%)
    }

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.search_agent = SearchAgent(api_key)
        self.add_sub_agent(self.search_agent)
        self._openai_client = None

    @property
    def openai_client(self):
        """Lazy OpenAI client initialization"""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai_client

    def match(
        self,
        hypotheses: List[Hypothesis],
        region: str = None,
        min_score: float = 0.3,  # 0-1 스케일
        top_k: int = 5
    ) -> Dict[str, List[MatchedEvent]]:
        """
        검증된 가설들에 대한 Event 매칭 (하이브리드 스코어링)

        Args:
            hypotheses: 검증된 가설 목록
            region: 지역 필터
            min_score: 최소 점수 (0-1 스케일, 기본 0.3)
            top_k: 가설당 최대 이벤트 수

        Returns:
            {hypothesis_id: [MatchedEvent, ...]}
        """
        results = {}

        for hypothesis in hypotheses:
            if not hypothesis.validated:
                continue

            matched_events = self._match_single(hypothesis, region, min_score, top_k)
            if matched_events:
                results[hypothesis.id] = matched_events

        return results

    def _match_single(
        self,
        hypothesis: Hypothesis,
        region: str = None,
        min_score: float = 0.3,
        top_k: int = 5
    ) -> List[MatchedEvent]:
        """단일 가설에 대한 하이브리드 매칭"""

        # 1. 가설 description을 embedding으로 변환
        hypothesis_embedding = self._get_embedding(hypothesis.description)

        # 2. Vector Search로 유사 Event 검색
        vector_results = self._vector_search(hypothesis_embedding, top_k=20)

        # 3. Graph Search로 Factor 관련 Event 검색
        graph_results = self._graph_search(hypothesis, region)

        # 4. 두 결과 병합 및 하이브리드 스코어 계산
        all_events = self._merge_and_score(
            hypothesis,
            hypothesis_embedding,
            vector_results,
            graph_results,
            region
        )

        # 5. 필터링 및 정렬
        filtered = [e for e in all_events if e.total_score >= min_score]
        filtered.sort(key=lambda x: x.total_score, reverse=True)

        return filtered[:top_k]

    def _get_embedding(self, text: str) -> List[float]:
        """텍스트를 embedding으로 변환"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding 생성 오류: {e}")
            return []

    def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 20
    ) -> List[Dict]:
        """Neo4j Vector Index로 유사 Event 검색"""
        if not query_embedding:
            return []

        query = """
        CALL db.index.vector.queryNodes('event_embedding', $top_k, $embedding)
        YIELD node, score
        MATCH (node)-[r:INCREASES|DECREASES]->(f:Factor)
        OPTIONAL MATCH (node)-[:TARGETS]->(d:Dimension)
        RETURN
            node.id as event_id,
            node.name as event_name,
            node.category as event_category,
            node.severity as event_severity,
            node.is_ongoing as is_ongoing,
            node.evidence as evidence,
            node.source_urls as source_urls,
            node.source_titles as source_titles,
            f.name as factor_name,
            type(r) as impact_type,
            r.magnitude as magnitude,
            score as vector_score,
            collect(DISTINCT d.name) as target_regions
        """

        params = {
            "embedding": query_embedding,
            "top_k": top_k
        }

        try:
            result = self.search_agent.graph_tool.execute(query, params)
            if result.success and result.data:
                return result.data
        except Exception as e:
            print(f"Vector Search 오류: {e}")

        return []

    def _graph_search(
        self,
        hypothesis: Hypothesis,
        region: str = None
    ) -> List[Dict]:
        """Knowledge Graph에서 Event → Factor → Anchor 전체 경로 검색"""

        # Factor 이름 추출 (가설에서)
        factor_keywords = self._extract_factor_keywords(hypothesis.factor)

        # 가설 카테고리에서 Anchor 추출
        anchor_mapping = {
            "revenue": "revenue",
            "cost": "cost",
            "pricing": "revenue",  # 가격 조건은 매출에 영향
            "external": None  # 외부 요인은 특정 Anchor 없음
        }
        target_anchor = anchor_mapping.get(hypothesis.category)

        # Anchor가 지정된 경우: Event → Factor → Anchor 전체 경로 검증
        if target_anchor:
            query = """
            MATCH (e:Event)-[r1:INCREASES|DECREASES]->(f:Factor)-[r2:AFFECTS]->(a:Anchor)
            WHERE any(kw IN $keywords WHERE toLower(f.name) CONTAINS toLower(kw))
              AND a.id = $anchor_id
            OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
            RETURN
                e.id as event_id,
                e.name as event_name,
                e.category as event_category,
                e.severity as event_severity,
                e.is_ongoing as is_ongoing,
                e.evidence as evidence,
                e.embedding as embedding,
                e.source_urls as source_urls,
                e.source_titles as source_titles,
                f.name as factor_name,
                type(r1) as impact_type,
                r1.magnitude as magnitude,
                r1.confidence as confidence,
                a.id as anchor_id,
                r2.type as factor_anchor_relation,
                collect(DISTINCT d.name) as target_regions
            LIMIT 30
            """
            params = {"keywords": factor_keywords, "anchor_id": target_anchor}
        else:
            # Anchor 지정 없는 경우: 기존 쿼리 (Event → Factor만)
            query = """
            MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor)
            WHERE any(kw IN $keywords WHERE toLower(f.name) CONTAINS toLower(kw))
            OPTIONAL MATCH (f)-[r2:AFFECTS]->(a:Anchor)
            OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
            RETURN
                e.id as event_id,
                e.name as event_name,
                e.category as event_category,
                e.severity as event_severity,
                e.is_ongoing as is_ongoing,
                e.evidence as evidence,
                e.embedding as embedding,
                e.source_urls as source_urls,
                e.source_titles as source_titles,
                f.name as factor_name,
                type(r) as impact_type,
                r.magnitude as magnitude,
                r.confidence as confidence,
                a.id as anchor_id,
                r2.type as factor_anchor_relation,
                collect(DISTINCT d.name) as target_regions
            LIMIT 30
            """
            params = {"keywords": factor_keywords}

        try:
            result = self.search_agent.graph_tool.execute(query, params)
            if result.success and result.data:
                return result.data
        except Exception as e:
            print(f"Graph Search 오류: {e}")

        return []

    def _extract_factor_keywords(self, factor: str) -> List[str]:
        """Factor에서 검색 키워드 추출"""
        # 기본 키워드
        keywords = [factor]

        # Factor 매핑 테이블
        mapping = {
            "물류비": ["물류비", "해상운임", "운송비", "운임"],
            "재료비": ["원재료비", "패널가격", "재료비"],
            "관세": ["관세", "tariff"],
            "환율": ["환율", "원달러"],
            "경쟁": ["경쟁심화", "경쟁"],
            "수요": ["수요부진", "수요", "판매"],
            "OLED": ["OLED", "올레드"],
        }

        for key, values in mapping.items():
            if key in factor:
                keywords.extend(values)
                break

        return list(set(keywords))

    def _merge_and_score(
        self,
        hypothesis: Hypothesis,
        hypothesis_embedding: List[float],
        vector_results: List[Dict],
        graph_results: List[Dict],
        region: str = None
    ) -> List[MatchedEvent]:
        """Vector + Graph 결과 병합 및 하이브리드 스코어 계산"""

        # event_id 기준으로 병합
        events_map = {}

        # Vector 결과 추가
        for vr in vector_results:
            event_id = vr.get("event_id", "")
            if event_id:
                events_map[event_id] = {
                    **vr,
                    "vector_score": vr.get("vector_score", 0)
                }

        # Graph 결과 병합
        for gr in graph_results:
            event_id = gr.get("event_id", "")
            if event_id:
                if event_id in events_map:
                    # 이미 있으면 graph 정보 추가
                    events_map[event_id]["from_graph"] = True
                    events_map[event_id]["confidence"] = gr.get("confidence", 0.8)
                    # source_urls, source_titles도 복사 (Vector 결과에 없을 수 있음)
                    if not events_map[event_id].get("source_urls"):
                        events_map[event_id]["source_urls"] = gr.get("source_urls", [])
                        events_map[event_id]["source_titles"] = gr.get("source_titles", [])
                else:
                    # 새로 추가
                    events_map[event_id] = {
                        **gr,
                        "vector_score": 0,
                        "from_graph": True
                    }

        # 각 이벤트에 대해 하이브리드 스코어 계산
        matched_events = []
        for event_id, event_data in events_map.items():
            score, breakdown = self._calculate_hybrid_score(
                hypothesis,
                hypothesis_embedding,
                event_data,
                region
            )

            # sources 구성 (source_urls, source_titles 배열에서)
            sources = event_data.get("sources", [])
            if not sources:
                source_urls = event_data.get("source_urls", []) or []
                source_titles = event_data.get("source_titles", []) or []
                event_name = event_data.get("event_name", "")

                for i, url in enumerate(source_urls[:3]):  # 최대 3개
                    if url:
                        title = source_titles[i] if i < len(source_titles) else event_name
                        sources.append({"url": url, "title": title, "link": url})

            matched_events.append(MatchedEvent(
                event_id=event_id,
                event_name=event_data.get("event_name", ""),
                event_category=event_data.get("event_category", ""),
                severity=event_data.get("event_severity", "medium"),
                is_ongoing=event_data.get("is_ongoing", False),
                matched_factor=event_data.get("factor_name", ""),
                impact_type=event_data.get("impact_type", ""),
                magnitude=event_data.get("magnitude", "medium"),
                total_score=score,
                score_breakdown=breakdown,
                sources=sources,
                evidence=event_data.get("evidence", ""),
                target_regions=event_data.get("target_regions", [])
            ))

        return matched_events

    def _calculate_hybrid_score(
        self,
        hypothesis: Hypothesis,
        hypothesis_embedding: List[float],
        event_data: Dict,
        region: str = None
    ) -> tuple:
        """
        하이브리드 스코어 계산

        Final Score = α * Semantic_Score + β * Graph_Score

        Semantic Score (0-1):
        - Neo4j Vector Search의 cosine similarity

        Graph Score (0-1):
        - Direction Match (0.4): 가설 방향과 Event 영향 일치
        - Magnitude (0.3): Event → Factor 관계 강도
        - Region Match (0.2): 지역 일치
        - Severity (0.1): 이벤트 심각도
        """
        breakdown = {}

        # === 1. Semantic Score (Vector Similarity) ===
        # Neo4j Vector Search가 이미 cosine similarity를 반환
        vector_score = event_data.get("vector_score", 0)

        # Vector search에서 안 온 경우 직접 계산
        if vector_score == 0 and hypothesis_embedding:
            event_embedding = event_data.get("embedding")
            if event_embedding:
                vector_score = self._cosine_similarity(hypothesis_embedding, event_embedding)

        breakdown["semantic"] = round(vector_score, 3)

        # === 2. Graph Score ===
        graph_score = 0.0

        # 2.1 Direction Match (40% of graph score)
        direction_score = self._calc_direction_score(hypothesis, event_data)
        breakdown["direction"] = round(direction_score, 3)

        # 2.2 Magnitude (30% of graph score)
        magnitude = event_data.get("magnitude", "medium")
        magnitude_score = MAGNITUDE_SCORES.get(magnitude, 0.5)
        breakdown["magnitude"] = round(magnitude_score, 3)

        # 2.3 Region Match (20% of graph score)
        region_score = self._calc_region_score(event_data, region)
        breakdown["region"] = round(region_score, 3)

        # 2.4 Severity (10% of graph score)
        severity = event_data.get("event_severity", "medium")
        severity_score = SEVERITY_SCORES.get(severity, 0.5)
        breakdown["severity"] = round(severity_score, 3)

        # Graph Score 합산 (가중 평균)
        graph_score = (
            direction_score * 0.4 +
            magnitude_score * 0.3 +
            region_score * 0.2 +
            severity_score * 0.1
        )
        breakdown["graph"] = round(graph_score, 3)

        # === 3. Final Hybrid Score ===
        semantic_weight = self.WEIGHTS["semantic"]
        graph_weight = self.WEIGHTS["graph"]

        final_score = (
            vector_score * semantic_weight +
            graph_score * graph_weight
        )

        breakdown["final"] = round(final_score, 3)

        return final_score, breakdown

    def _calc_direction_score(self, hypothesis: Hypothesis, event_data: Dict) -> float:
        """방향 일치 점수 계산"""
        event_impact = event_data.get("impact_type", "").upper()
        hypothesis_direction = hypothesis.direction.lower()

        # 부정적 Factor 체크 ("부진", "위축" 등)
        factor_name = event_data.get("factor_name", "").lower()
        is_negative = any(kw in factor_name for kw in ["부진", "위축", "감소", "하락"])

        if hypothesis_direction == "increase":
            if event_impact == "INCREASES":
                return 0.0 if is_negative else 1.0
            elif event_impact == "DECREASES":
                return 1.0 if is_negative else 0.0
        elif hypothesis_direction == "decrease":
            if event_impact == "DECREASES":
                return 0.0 if is_negative else 1.0
            elif event_impact == "INCREASES":
                return 1.0 if is_negative else 0.0

        return 0.5  # 불확실한 경우

    def _calc_region_score(self, event_data: Dict, region: str) -> float:
        """지역 일치 점수 계산"""
        if not region:
            return 0.7  # 지역 필터 없음

        target_regions = event_data.get("target_regions", [])
        if not target_regions:
            return 0.5  # 글로벌 이벤트

        normalized_region = self._normalize_region(region)
        if normalized_region in target_regions:
            return 1.0  # 정확히 일치

        if "global" in [r.lower() for r in target_regions]:
            return 0.7  # 글로벌 이벤트

        return 0.0  # 불일치

    def _normalize_region(self, region: str) -> Optional[str]:
        """지역 코드 정규화"""
        if not region:
            return None

        region_upper = region.upper()
        mapping = {
            "NA": "NA", "NORTH AMERICA": "NA", "북미": "NA",
            "EU": "EU", "EUROPE": "EU", "유럽": "EU",
            "KR": "KR", "KOREA": "KR", "한국": "KR",
            "ASIA": "ASIA", "아시아": "ASIA"
        }
        return mapping.get(region_upper, region_upper)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def run(self, context: AgentContext) -> Dict[str, Any]:
        """Agent 실행"""
        hypotheses = context.metadata.get("validated_hypotheses", [])
        region = context.metadata.get("region")
        min_score = context.metadata.get("min_score", 0.3)
        top_k = context.metadata.get("top_k", 5)

        matched = self.match(
            hypotheses=hypotheses,
            region=region,
            min_score=min_score,
            top_k=top_k
        )

        # MatchedEvent를 직렬화 가능한 형태로 변환
        serialized = {}
        for h_id, events in matched.items():
            serialized[h_id] = [
                {
                    "event_name": ev.event_name,
                    "event_category": ev.event_category,
                    "severity": ev.severity,
                    "impact_type": ev.impact_type,
                    "total_score": ev.total_score,
                    "score_breakdown": ev.score_breakdown,
                    "sources": ev.sources[:2],
                    "evidence": ev.evidence[:200] if ev.evidence else ""
                }
                for ev in events
            ]

        result = {
            "matched_events": matched,
            "matched_serialized": serialized,
            "hypothesis_count": len(matched),
            "total_events": sum(len(v) for v in matched.values())
        }

        context.add_step("event_matching", {
            "hypothesis_count": len(matched),
            "total_events": sum(len(v) for v in matched.values())
        })

        return result
