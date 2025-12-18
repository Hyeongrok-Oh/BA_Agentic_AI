"""Layer 2 데이터 모델 - Factor와 Anchor 관계"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import date


class FactorCategory(Enum):
    """Factor 카테고리"""
    MACRO_ECONOMY = "macro_economy"      # 환율, 금리, 경기
    DEMAND = "demand"                     # 수요, 소비심리
    COMPETITION = "competition"           # 경쟁, 점유율
    SUPPLY_CHAIN = "supply_chain"         # 패널가격, 물류비
    TRADE = "trade"                       # 관세, 무역
    TECHNOLOGY = "technology"             # 프리미엄화, 혁신


class RelationType(Enum):
    """Factor → Anchor 관계 타입"""
    PROPORTIONAL = "PROPORTIONAL"                    # 정비례 (Factor↑ → Anchor↑)
    INVERSELY_PROPORTIONAL = "INVERSELY_PROPORTIONAL"  # 반비례 (Factor↑ → Anchor↓)


@dataclass
class SourceReference:
    """문서 출처 정보"""
    doc_name: str           # 파일명
    doc_date: date          # 문서 날짜
    doc_type: str           # consensus / dart
    paragraph: str          # 원문 문단
    page_num: Optional[int] = None


@dataclass
class FactorMention:
    """Factor 언급 정보"""
    factor_name: str
    anchor_id: str          # revenue, quantity, cost
    relation_type: RelationType
    source: SourceReference
    confidence: float = 1.0  # 추출 신뢰도 (0~1)


@dataclass
class FactorNode:
    """Factor 노드"""
    id: str
    name: str
    category: FactorCategory
    mentions: List[FactorMention] = field(default_factory=list)

    @property
    def mention_count(self) -> int:
        return len(self.mentions)

    @property
    def sources(self) -> List[str]:
        return list(set(m.source.doc_name for m in self.mentions))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "mention_count": self.mention_count,
            "sources": self.sources,
        }


@dataclass
class FactorAnchorRelation:
    """Factor-Anchor 관계 (집계)"""
    factor_id: str
    anchor_id: str
    relation_type: RelationType
    mention_count: int
    sources: List[SourceReference]

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "anchor_id": self.anchor_id,
            "relation_type": self.relation_type.value,
            "mention_count": self.mention_count,
            "source_count": len(self.sources),
        }


@dataclass
class Layer2Graph:
    """Layer 2 그래프 데이터"""
    factors: List[FactorNode] = field(default_factory=list)
    relations: List[FactorAnchorRelation] = field(default_factory=list)

    def add_mention(self, mention: FactorMention) -> None:
        """Factor 언급 추가"""
        # Factor 찾기 또는 생성
        factor = self._get_or_create_factor(mention.factor_name)
        factor.mentions.append(mention)

    def _get_or_create_factor(self, name: str) -> FactorNode:
        """Factor 노드 조회 또는 생성"""
        factor_id = self._name_to_id(name)
        for f in self.factors:
            if f.id == factor_id:
                return f

        # 새로 생성
        new_factor = FactorNode(
            id=factor_id,
            name=name,
            category=FactorCategory.MACRO_ECONOMY,  # 기본값, 나중에 분류
        )
        self.factors.append(new_factor)
        return new_factor

    def _name_to_id(self, name: str) -> str:
        """이름을 ID로 변환"""
        return name.lower().replace(" ", "_").replace("/", "_")

    def aggregate_relations(self) -> None:
        """Factor-Anchor 관계 집계"""
        relation_map = {}

        for factor in self.factors:
            for mention in factor.mentions:
                key = (factor.id, mention.anchor_id, mention.relation_type)
                if key not in relation_map:
                    relation_map[key] = {
                        "factor_id": factor.id,
                        "anchor_id": mention.anchor_id,
                        "relation_type": mention.relation_type,
                        "sources": [],
                    }
                relation_map[key]["sources"].append(mention.source)

        self.relations = [
            FactorAnchorRelation(
                factor_id=v["factor_id"],
                anchor_id=v["anchor_id"],
                relation_type=v["relation_type"],
                mention_count=len(v["sources"]),
                sources=v["sources"],
            )
            for v in relation_map.values()
        ]

    def summary(self) -> dict:
        return {
            "factors": len(self.factors),
            "relations": len(self.relations),
            "total_mentions": sum(f.mention_count for f in self.factors),
        }
