"""Event 추출 - LLM 기반 Event-Factor 관계 추출"""

import json
import os
import re
from typing import List, Optional
from datetime import datetime, date
from dataclasses import dataclass

from .models import (
    EventNode, EventFactorRelation, EventDimensionRelation,
    EventSource, EventCategory, ImpactType, Severity, Layer3Graph
)
from .search_client import SearchResult
from .config import CORE_FACTORS, DIMENSIONS


# Event 추출 프롬프트
EVENT_EXTRACTION_PROMPT = """다음 뉴스 검색 결과에서 LG전자 TV(HE) 사업에 영향을 미치는 **구체적인 이벤트**를 추출하세요.

**검색 결과:**
{search_results}

**추출 규칙:**
1. **이벤트**: 구체적이고 시간이 특정된 사건만 추출 (일반 트렌드/상태 제외)
   - O: "홍해 사태로 컨테이너 운임 3배 급등", "트럼프 25% 보편관세 발표"
   - X: "경쟁 심화", "수요 부진" (이것은 Factor)
2. TV/가전 사업에 직접/간접적으로 영향이 있는 이벤트만
3. 각 이벤트가 어떤 Factor에 영향을 주는지 판단
4. **중요**: 각 이벤트의 출처가 된 검색 결과 번호를 source_indices에 명시

**영향 받는 Factor 후보:**
{factors}

**타겟 Dimension 후보:**
- Region: NA(북미), EU(유럽), ASIA(아시아), KR(한국)
- ProductCategory: OLED, QNED, LCD

**응답 형식 (JSON):**
```json
{{
  "events": [
    {{
      "name": "이벤트명 (한글)",
      "name_en": "Event name (English)",
      "category": "geopolitical|policy|market|company|macro_economy|technology|natural",
      "start_date": "YYYY-MM-DD 또는 null",
      "is_ongoing": true/false,
      "severity": "low|medium|high|critical",
      "source_indices": [1, 3],
      "affected_factors": [
        {{"factor": "Factor명", "impact": "INCREASES|DECREASES", "magnitude": "low|medium|high"}}
      ],
      "target_dimensions": [
        {{"dimension": "NA|EU|ASIA|KR|OLED|QNED|LCD", "type": "Region|ProductCategory"}}
      ],
      "evidence": "영향 근거 (뉴스 snippet에서 발췌)"
    }}
  ]
}}
```

**source_indices**: 이 이벤트를 추출한 검색 결과 번호 (위의 [1], [2], ... 참조)

이벤트가 없으면: {{"events": []}}
"""


@dataclass
class ExtractedEvent:
    """추출된 이벤트"""
    name: str
    name_en: Optional[str]
    category: str
    start_date: Optional[str]
    is_ongoing: bool
    severity: str
    source_indices: List[int]  # 출처 검색결과 인덱스
    affected_factors: List[dict]
    target_dimensions: List[dict]
    evidence: str


class EventExtractor:
    """LLM 기반 Event 추출"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 필요")

    def extract_from_results(
        self,
        search_results: List[SearchResult],
        batch_size: int = 10
    ) -> List[ExtractedEvent]:
        """검색 결과에서 Event 추출"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지 필요: pip install openai")

        client = OpenAI(api_key=self.api_key)

        # 검색 결과 포맷팅
        formatted_results = self._format_results(search_results[:batch_size])
        factors_str = ", ".join(CORE_FACTORS)

        prompt = EVENT_EXTRACTION_PROMPT.format(
            search_results=formatted_results,
            factors=factors_str
        )

        message = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.choices[0].message.content
        return self._parse_response(response_text)

    def _format_results(self, results: List[SearchResult]) -> str:
        """검색 결과 포맷팅"""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.title}")
            lines.append(f"    날짜: {r.date or 'N/A'} | 출처: {r.source or 'N/A'}")
            lines.append(f"    내용: {r.snippet}")
            lines.append("")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> List[ExtractedEvent]:
        """LLM 응답 파싱"""
        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 직접 JSON 객체 찾기
            json_match2 = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match2:
                json_str = json_match2.group(0)
            else:
                json_str = response

        # JSON 정리 (LLM 출력 오류 수정)
        json_str = self._clean_json(json_str)

        try:
            data = json.loads(json_str)
            events = data.get("events", [])
            return [
                ExtractedEvent(
                    name=e.get("name", ""),
                    name_en=e.get("name_en"),
                    category=e.get("category", "market"),
                    start_date=e.get("start_date"),
                    is_ongoing=e.get("is_ongoing", False),
                    severity=e.get("severity", "medium"),
                    source_indices=e.get("source_indices", []),
                    affected_factors=e.get("affected_factors", []),
                    target_dimensions=e.get("target_dimensions", []),
                    evidence=e.get("evidence", "")
                )
                for e in events
            ]
        except json.JSONDecodeError as ex:
            print(f"JSON 파싱 오류: {ex}")
            print(f"JSON 내용 (처음 300자): {json_str[:300]}")
            return []

    def _clean_json(self, json_str: str) -> str:
        """LLM 출력 JSON 정리"""
        # Trailing comma 제거: }, ] 또는 }, } 앞의 쉼표
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        # null 값 처리
        json_str = json_str.replace(': null', ': null')
        return json_str


class Layer3Builder:
    """Layer 3 그래프 빌더"""

    def __init__(self):
        self.extractor = EventExtractor()
        self.graph = Layer3Graph()
        self._event_id_counter = 0

    def build_from_search_results(
        self,
        search_results: List[SearchResult],
        batch_size: int = 10,
        verbose: bool = True
    ) -> Layer3Graph:
        """검색 결과에서 Event 추출 및 그래프 구축"""
        total = len(search_results)
        processed = 0

        if verbose:
            print(f"=== Event 추출 시작 ===")
            print(f"총 검색 결과: {total}개")
            print(f"배치 크기: {batch_size}개")

        # 배치 단위로 처리
        for i in range(0, total, batch_size):
            batch = search_results[i:i + batch_size]
            processed += len(batch)

            try:
                extracted = self.extractor.extract_from_results(batch, batch_size)

                for event_data in extracted:
                    event = self._create_event_node(event_data, batch)
                    self.graph.add_event(event)

                if verbose:
                    print(f"  진행: {processed}/{total} - 추출: {len(extracted)}개 이벤트")

            except Exception as e:
                print(f"  추출 오류: {e}")
                continue

        if verbose:
            print(f"\n=== 추출 완료 ===")
            print(f"총 Event: {len(self.graph.events)}개")
            summary = self.graph.summary()
            print(f"Factor 관계: {summary['total_factor_relations']}개")
            print(f"Dimension 관계: {summary['total_dimension_relations']}개")

        return self.graph

    def _create_event_node(
        self,
        event_data: ExtractedEvent,
        source_results: List[SearchResult]
    ) -> EventNode:
        """ExtractedEvent를 EventNode로 변환"""
        # ID 생성
        self._event_id_counter += 1
        event_id = self._generate_event_id(event_data.name)

        # 카테고리 파싱
        try:
            category = EventCategory(event_data.category)
        except ValueError:
            category = EventCategory.MARKET

        # 심각도 파싱
        try:
            severity = Severity(event_data.severity)
        except ValueError:
            severity = Severity.MEDIUM

        # 날짜 파싱
        start_date = None
        if event_data.start_date:
            try:
                start_date = datetime.strptime(event_data.start_date, "%Y-%m-%d").date()
            except ValueError:
                pass

        # Factor 관계 생성
        factor_relations = []
        for f in event_data.affected_factors:
            factor_name = f.get("factor", "")
            factor_id = factor_name.lower().replace(" ", "_")
            try:
                impact = ImpactType(f.get("impact", "INCREASES"))
            except ValueError:
                impact = ImpactType.INCREASES

            factor_relations.append(EventFactorRelation(
                factor_name=factor_name,
                factor_id=factor_id,
                impact_type=impact,
                magnitude=f.get("magnitude", "medium"),
                evidence=event_data.evidence
            ))

        # Dimension 관계 생성
        dimension_relations = []
        for d in event_data.target_dimensions:
            dimension_relations.append(EventDimensionRelation(
                dimension_name=d.get("dimension", ""),
                dimension_type=d.get("type", "Region")
            ))

        # Source 생성 - LLM이 지정한 source_indices 사용
        sources = []
        for idx in event_data.source_indices:
            # 인덱스는 1부터 시작 (LLM 프롬프트에서 [1], [2]로 표시)
            actual_idx = idx - 1
            if 0 <= actual_idx < len(source_results):
                r = source_results[actual_idx]
                sources.append(EventSource(
                    url=r.link,
                    title=r.title,
                    snippet=r.snippet,
                    published_date=None,
                    source_name=r.source,
                    search_query=r.query
                ))

        # source_indices가 비어있으면 fallback (첫 번째 결과)
        if not sources and source_results:
            r = source_results[0]
            sources.append(EventSource(
                url=r.link,
                title=r.title,
                snippet=r.snippet,
                published_date=None,
                source_name=r.source,
                search_query=r.query
            ))

        return EventNode(
            id=event_id,
            name=event_data.name,
            name_en=event_data.name_en,
            category=category,
            start_date=start_date,
            is_ongoing=event_data.is_ongoing,
            severity=severity,
            sources=sources,
            factor_relations=factor_relations,
            dimension_relations=dimension_relations,
            evidence=event_data.evidence
        )

    def _generate_event_id(self, name: str) -> str:
        """Event ID 생성"""
        # 한글/영어 이름에서 ID 생성
        clean_name = re.sub(r'[^\w\s]', '', name.lower())
        clean_name = clean_name.replace(" ", "_")[:30]
        return f"event_{clean_name}_{self._event_id_counter}"


def test_extraction():
    """추출 테스트"""
    from .search_client import BraveSearchClient

    print("=== Event 추출 테스트 ===\n")

    # 검색
    client = BraveSearchClient()
    results = client.search_news("홍해 사태 해운", count=10)
    print(f"검색 결과: {len(results)}개\n")

    # 추출
    builder = Layer3Builder()
    graph = builder.build_from_search_results(results, batch_size=10)

    print(f"\n추출된 Event:")
    for event in graph.events:
        print(f"- {event.name}")
        for r in event.factor_relations:
            print(f"  → {r.impact_type.value} {r.factor_name}")


if __name__ == "__main__":
    test_extraction()
