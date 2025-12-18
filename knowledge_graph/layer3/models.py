"""Layer 3 데이터 모델 - Event와 Factor/Dimension 관계"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import date, datetime


class EventCategory(Enum):
    """Event 카테고리"""
    GEOPOLITICAL = "geopolitical"      # 지정학적 (홍해 사태, 전쟁)
    POLICY = "policy"                   # 정책/규제 (관세, 환경 규제)
    MARKET = "market"                   # 시장 (패널 가격, 유가)
    COMPANY = "company"                 # 기업 (실적 발표, 신제품)
    MACRO_ECONOMY = "macro_economy"     # 거시경제 (금리, 환율)
    TECHNOLOGY = "technology"           # 기술 (신기술, AI)
    NATURAL = "natural"                 # 자연재해/팬데믹


class ImpactType(Enum):
    """Event → Factor 영향 타입"""
    INCREASES = "INCREASES"     # Event가 Factor를 증가시킴
    DECREASES = "DECREASES"     # Event가 Factor를 감소시킴


class Severity(Enum):
    """Event 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EventSource:
    """뉴스 출처 정보"""
    url: str
    title: str
    snippet: str
    published_date: Optional[date] = None
    source_name: Optional[str] = None  # 뉴스 매체명
    search_query: Optional[str] = None  # 검색 쿼리


@dataclass
class EventFactorRelation:
    """Event → Factor 관계"""
    factor_name: str
    factor_id: str
    impact_type: ImpactType
    magnitude: str = "medium"  # low, medium, high
    confidence: float = 0.8
    evidence: str = ""


@dataclass
class EventDimensionRelation:
    """Event → Dimension 관계"""
    dimension_name: str
    dimension_type: str  # Region, ProductCategory
    specificity: str = "medium"  # low, medium, high


@dataclass
class EventNode:
    """Event 노드"""
    id: str
    name: str
    name_en: Optional[str] = None
    category: EventCategory = EventCategory.MARKET
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_ongoing: bool = False
    severity: Severity = Severity.MEDIUM
    region_scope: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    sources: List[EventSource] = field(default_factory=list)
    factor_relations: List[EventFactorRelation] = field(default_factory=list)
    dimension_relations: List[EventDimensionRelation] = field(default_factory=list)
    evidence: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "name_en": self.name_en,
            "category": self.category.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_ongoing": self.is_ongoing,
            "severity": self.severity.value,
            "region_scope": self.region_scope,
            "aliases": self.aliases,
            "source_count": self.source_count,
            "sources": [
                {
                    "url": s.url,
                    "title": s.title,
                    "snippet": s.snippet,
                    "source_name": s.source_name,
                }
                for s in self.sources
            ],
            "factor_relations": [
                {
                    "factor": r.factor_name,
                    "impact": r.impact_type.value,
                    "magnitude": r.magnitude,
                }
                for r in self.factor_relations
            ],
            "dimension_relations": [
                {
                    "dimension": r.dimension_name,
                    "type": r.dimension_type,
                }
                for r in self.dimension_relations
            ],
            "evidence": self.evidence,
        }


@dataclass
class EventChunk:
    """Event 콘텐츠 청크 (Vector 저장용)"""
    event_id: str
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "has_embedding": self.embedding is not None,
            "metadata": self.metadata,
        }


@dataclass
class Layer3Graph:
    """Layer 3 그래프 데이터"""
    events: List[EventNode] = field(default_factory=list)
    chunks: List[EventChunk] = field(default_factory=list)

    def add_event(self, event: EventNode) -> None:
        """Event 추가 (중복 체크)"""
        # 중복 체크
        for existing in self.events:
            if existing.id == event.id:
                # 기존 이벤트에 소스 병합
                existing.sources.extend(event.sources)
                return
        self.events.append(event)

    def get_event_by_id(self, event_id: str) -> Optional[EventNode]:
        """ID로 Event 조회"""
        for event in self.events:
            if event.id == event_id:
                return event
        return None

    def get_events_by_category(self, category: EventCategory) -> List[EventNode]:
        """카테고리별 Event 조회"""
        return [e for e in self.events if e.category == category]

    def get_events_affecting_factor(self, factor_name: str) -> List[EventNode]:
        """특정 Factor에 영향을 주는 Event 조회"""
        return [
            e for e in self.events
            if any(r.factor_name == factor_name for r in e.factor_relations)
        ]

    def summary(self) -> dict:
        """요약 통계"""
        category_counts = {}
        for e in self.events:
            cat = e.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        impact_counts = {"INCREASES": 0, "DECREASES": 0}
        for e in self.events:
            for r in e.factor_relations:
                impact_counts[r.impact_type.value] += 1

        return {
            "total_events": len(self.events),
            "total_chunks": len(self.chunks),
            "by_category": category_counts,
            "impact_counts": impact_counts,
            "total_factor_relations": sum(len(e.factor_relations) for e in self.events),
            "total_dimension_relations": sum(len(e.dimension_relations) for e in self.events),
        }
