"""
Event Matcher Agent - 하이브리드 스코어링 (Vector + Graph)

개선된 버전:
- Factor-Event 직접 매칭 강화
- 가설별 고유 이벤트 우선
- 중복 이벤트 페널티
"""

import os
from typing import Dict, Any, List, Optional, Set
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

# Factor → 관련 키워드 매핑 (정밀하게)
FACTOR_KEYWORD_MAP = {
    # 물류/운송
    "물류비": ["물류비", "물류"],
    "해상운임": ["해상운임", "운임", "해운"],
    "운송비": ["운송비", "운송"],
    # 원가
    "원재료비": ["원재료", "원자재", "재료비"],
    "패널가격": ["패널가격", "패널", "디스플레이"],
    "부품비": ["부품비", "부품"],
    # 가격/관세
    "관세": ["관세", "tariff", "section 232"],
    "환율": ["환율", "원달러", "달러"],
    "가격경쟁력": ["가격경쟁", "가격"],
    # 수요
    "TV수요": ["TV수요", "TV 수요", "가전수요"],
    "수요부진": ["수요부진", "수요 부진", "소비위축"],
    "소비심리": ["소비심리", "소비자심리"],
    # 경쟁
    "경쟁심화": ["경쟁심화", "경쟁 심화", "시장경쟁"],
    "시장점유율": ["점유율", "시장점유"],
    # 제품
    "OLED": ["OLED", "올레드"],
    "QNED": ["QNED", "퀴네드"],
    "프리미엄": ["프리미엄", "고급"],
    # 실적
    "실적": ["실적", "영업이익", "매출"],
    "수익성": ["수익성", "마진", "이익률"],
}


class EventMatcher(BaseAgent):
    """개선된 하이브리드 이벤트 매칭 에이전트"""

    name = "event_matcher"
    description = "Factor-Event 직접 매칭 + Vector Similarity 기반 하이브리드 스코어링"

    # 개선된 스코어 가중치 (Graph 비중 증가)
    WEIGHTS = {
        "semantic": 0.2,      # Vector Similarity (20% ← 40%에서 감소)
        "graph": 0.5,         # Graph Score (50%)
        "factor_match": 0.3,  # Factor 직접 매칭 (30% 신규)
    }

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.search_agent = SearchAgent(api_key)
        self.add_sub_agent(self.search_agent)
        self._openai_client = None
        self._used_events: Set[str] = set()  # 이미 사용된 이벤트 추적

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
        min_score: float = 0.3,
        top_k: int = 5
    ) -> Dict[str, List[MatchedEvent]]:
        """
        검증된 가설들에 대한 Event 매칭 (개선된 하이브리드 스코어링)

        개선사항:
        1. Factor-Event 직접 매칭 우선
        2. 가설별 고유 이벤트 우선 (중복 페널티)
        3. 이미 사용된 이벤트 추적
        """
        results = {}
        self._used_events = set()  # 초기화

        for hypothesis in hypotheses:
            if not hypothesis.validated:
                continue

            matched_events = self._match_single(hypothesis, region, min_score, top_k)
            if matched_events:
                results[hypothesis.id] = matched_events

                # 사용된 이벤트 기록
                for ev in matched_events:
                    self._used_events.add(ev.event_id)

        return results

    def _match_single(
        self,
        hypothesis: Hypothesis,
        region: str = None,
        min_score: float = 0.3,
        top_k: int = 5
    ) -> List[MatchedEvent]:
        """단일 가설에 대한 개선된 매칭"""

        # 1. Factor 키워드 추출 (정밀)
        factor_keywords = self._extract_factor_keywords_precise(hypothesis.factor)

        # 2. Graph Search: Factor와 직접 연결된 Event만 검색 (우선)
        graph_results = self._graph_search_direct(hypothesis, factor_keywords, region)

        # 3. Vector Search: 보조적으로 사용 (Graph에서 못 찾은 경우)
        vector_results = []
        if len(graph_results) < top_k:
            hypothesis_embedding = self._get_embedding(hypothesis.description)
            vector_results = self._vector_search(hypothesis_embedding, top_k=15, region=region)

        # 4. 병합 및 스코어 계산
        all_events = self._merge_and_score_improved(
            hypothesis,
            factor_keywords,
            graph_results,
            vector_results,
            region
        )

        # 5. 필터링, 중복 페널티 적용, 정렬
        scored_events = self._apply_uniqueness_bonus(all_events)
        filtered = [e for e in scored_events if e.total_score >= min_score]
        filtered.sort(key=lambda x: x.total_score, reverse=True)

        return filtered[:top_k]

    def _extract_factor_keywords_precise(self, factor: str) -> List[str]:
        """Factor에서 정밀한 검색 키워드 추출"""
        keywords = []
        factor_lower = factor.lower()

        # 정확한 매핑 우선
        for key, values in FACTOR_KEYWORD_MAP.items():
            if key.lower() in factor_lower or factor_lower in key.lower():
                keywords.extend(values)

        # 매핑에 없으면 원본 사용
        if not keywords:
            keywords = [factor]

        # 중복 제거
        return list(set(keywords))

    def _graph_search_direct(
        self,
        hypothesis: Hypothesis,
        factor_keywords: List[str],
        region: str = None
    ) -> List[Dict]:
        """Factor와 직접 연결된 Event만 검색 (정밀)"""

        # 지역 필터 조건 - Region 노드 직접 확인
        region_filter = ""
        if region:
            normalized = self._normalize_region(region)
            if normalized:
                # Region이 없거나(global), 해당 region을 타겟하거나, 'global' 타겟인 경우
                region_filter = f"AND (size(target_regions) = 0 OR '{normalized}' IN target_regions OR 'global' IN [reg IN target_regions | toLower(reg)])"

        query = f"""
        MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor)
        WHERE any(kw IN $keywords WHERE toLower(f.name) CONTAINS toLower(kw))
        // Region 노드만 명시적으로 조회 (Layer 1 Dimension)
        OPTIONAL MATCH (e)-[:TARGETS]->(reg:Region)
        WITH e, r, f, collect(DISTINCT reg.id) as target_regions
        WHERE true {region_filter}
        RETURN
            e.id as event_id,
            e.name as event_name,
            e.category as event_category,
            e.severity as event_severity,
            e.is_ongoing as is_ongoing,
            e.evidence as evidence,
            e.source_urls as source_urls,
            e.source_titles as source_titles,
            f.name as factor_name,
            f.id as factor_id,
            type(r) as impact_type,
            r.magnitude as magnitude,
            r.confidence as confidence,
            target_regions,
            1.0 as factor_match_score
        ORDER BY
            CASE e.severity
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                ELSE 4
            END,
            CASE r.magnitude
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                ELSE 3
            END
        LIMIT 20
        """

        params = {"keywords": factor_keywords}

        try:
            result = self.search_agent.graph_tool.execute(query, params)
            if result.success and result.data:
                return result.data
        except Exception as e:
            print(f"Graph Search Direct 오류: {e}")

        return []

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
        top_k: int = 15,
        region: str = None
    ) -> List[Dict]:
        """Neo4j Vector Index로 유사 Event 검색 (보조)"""
        if not query_embedding:
            return []

        # Region 필터 조건
        region_filter = ""
        if region:
            normalized = self._normalize_region(region)
            if normalized:
                region_filter = f"WHERE size(target_regions) = 0 OR '{normalized}' IN target_regions OR 'global' IN [reg IN target_regions | toLower(reg)]"

        query = f"""
        CALL db.index.vector.queryNodes('event_embedding', $top_k, $embedding)
        YIELD node, score
        MATCH (node)-[r:INCREASES|DECREASES]->(f:Factor)
        // Region 노드만 명시적으로 조회 (Layer 1 Dimension)
        OPTIONAL MATCH (node)-[:TARGETS]->(reg:Region)
        WITH node, score, r, f, collect(DISTINCT reg.id) as target_regions
        {region_filter}
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
            target_regions
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

    def _merge_and_score_improved(
        self,
        hypothesis: Hypothesis,
        factor_keywords: List[str],
        graph_results: List[Dict],
        vector_results: List[Dict],
        region: str = None
    ) -> List[MatchedEvent]:
        """개선된 병합 및 스코어 계산"""

        events_map = {}

        # Graph 결과 추가 (우선)
        for gr in graph_results:
            event_id = gr.get("event_id", "")
            if event_id:
                events_map[event_id] = {
                    **gr,
                    "from_graph": True,
                    "factor_match_score": 1.0,  # 직접 매칭
                    "vector_score": 0
                }

        # Vector 결과 추가 (Graph에 없는 것만)
        for vr in vector_results:
            event_id = vr.get("event_id", "")
            if event_id and event_id not in events_map:
                # Factor 키워드 매칭 확인
                factor_name = vr.get("factor_name", "").lower()
                factor_match = any(kw.lower() in factor_name for kw in factor_keywords)

                events_map[event_id] = {
                    **vr,
                    "from_graph": False,
                    "factor_match_score": 0.7 if factor_match else 0.2,
                    "vector_score": vr.get("vector_score", 0)
                }

        # 각 이벤트에 대해 스코어 계산
        matched_events = []
        for event_id, event_data in events_map.items():
            score, breakdown = self._calculate_improved_score(
                hypothesis,
                event_data,
                region
            )

            # sources 구성
            sources = self._build_sources(event_data)

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

    def _calculate_improved_score(
        self,
        hypothesis: Hypothesis,
        event_data: Dict,
        region: str = None
    ) -> tuple:
        """
        개선된 스코어 계산

        Final Score = 0.2 × Semantic + 0.5 × Graph + 0.3 × Factor_Match

        Graph Score = 0.4 × Direction + 0.3 × Magnitude + 0.2 × Region + 0.1 × Severity
        """
        breakdown = {}

        # === 1. Semantic Score (20%) ===
        vector_score = event_data.get("vector_score", 0)
        breakdown["semantic"] = round(vector_score, 3)

        # === 2. Factor Match Score (30%) - 신규 ===
        factor_match_score = event_data.get("factor_match_score", 0)
        breakdown["factor_match"] = round(factor_match_score, 3)

        # === 3. Graph Score (50%) ===
        # Direction Match (40% of graph)
        direction_score = self._calc_direction_score(hypothesis, event_data)
        breakdown["direction"] = round(direction_score, 3)

        # Magnitude (30% of graph)
        magnitude = event_data.get("magnitude", "medium")
        magnitude_score = MAGNITUDE_SCORES.get(magnitude, 0.5)
        breakdown["magnitude"] = round(magnitude_score, 3)

        # Region Match (20% of graph)
        region_score = self._calc_region_score(event_data, region)
        breakdown["region"] = round(region_score, 3)

        # Severity (10% of graph)
        severity = event_data.get("event_severity", "medium")
        severity_score = SEVERITY_SCORES.get(severity, 0.5)
        breakdown["severity"] = round(severity_score, 3)

        # Graph Score 합산
        graph_score = (
            direction_score * 0.4 +
            magnitude_score * 0.3 +
            region_score * 0.2 +
            severity_score * 0.1
        )
        breakdown["graph"] = round(graph_score, 3)

        # === 4. Final Score ===
        final_score = (
            vector_score * self.WEIGHTS["semantic"] +
            graph_score * self.WEIGHTS["graph"] +
            factor_match_score * self.WEIGHTS["factor_match"]
        )

        breakdown["final"] = round(final_score, 3)

        return final_score, breakdown

    def _apply_uniqueness_bonus(self, events: List[MatchedEvent]) -> List[MatchedEvent]:
        """
        이미 다른 가설에서 사용된 이벤트에 페널티 적용
        새로운 이벤트에 보너스 적용
        """
        for event in events:
            if event.event_id in self._used_events:
                # 중복 이벤트: 15% 페널티
                event.total_score *= 0.85
                event.score_breakdown["uniqueness_penalty"] = -0.15
            else:
                # 고유 이벤트: 10% 보너스
                event.total_score *= 1.10
                event.total_score = min(event.total_score, 1.0)  # 최대 1.0
                event.score_breakdown["uniqueness_bonus"] = 0.10

        return events

    def _calc_direction_score(self, hypothesis: Hypothesis, event_data: Dict) -> float:
        """방향 일치 점수 계산"""
        event_impact = event_data.get("impact_type", "").upper()
        hypothesis_direction = hypothesis.direction.lower()

        # 부정적 Factor 체크
        factor_name = event_data.get("factor_name", "").lower()
        negative_keywords = ["부진", "위축", "감소", "하락", "손실", "적자"]
        is_negative_factor = any(kw in factor_name for kw in negative_keywords)

        if hypothesis_direction == "increase":
            if event_impact == "INCREASES":
                return 0.0 if is_negative_factor else 1.0
            elif event_impact == "DECREASES":
                return 1.0 if is_negative_factor else 0.0
        elif hypothesis_direction == "decrease":
            if event_impact == "DECREASES":
                return 0.0 if is_negative_factor else 1.0
            elif event_impact == "INCREASES":
                return 1.0 if is_negative_factor else 0.0

        return 0.5

    def _calc_region_score(self, event_data: Dict, region: str) -> float:
        """지역 일치 점수 계산"""
        if not region:
            return 0.7

        target_regions = event_data.get("target_regions", [])
        if not target_regions:
            return 0.5  # 글로벌/미지정

        normalized_region = self._normalize_region(region)
        target_regions_upper = [r.upper() if r else "" for r in target_regions]

        if normalized_region in target_regions_upper:
            return 1.0

        if "GLOBAL" in target_regions_upper or "전체" in target_regions:
            return 0.7

        return 0.2  # 불일치

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

    def _build_sources(self, event_data: Dict) -> List[Dict]:
        """출처 정보 구성"""
        sources = []
        source_urls = event_data.get("source_urls", []) or []
        source_titles = event_data.get("source_titles", []) or []
        event_name = event_data.get("event_name", "")

        for i, url in enumerate(source_urls[:3]):
            if url:
                title = source_titles[i] if i < len(source_titles) else event_name
                sources.append({"url": url, "title": title, "link": url})

        return sources

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
                    "matched_factor": ev.matched_factor,
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
