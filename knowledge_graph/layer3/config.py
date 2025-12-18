"""Layer 3 설정"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Layer3Config:
    """Layer 3 설정"""
    # API Keys
    brave_api_key: str = field(default_factory=lambda: os.getenv("BRAVE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Search Settings
    max_results_per_query: int = 20
    search_languages: List[str] = field(default_factory=lambda: ["ko", "en"])
    search_regions: List[str] = field(default_factory=lambda: ["kr", "us"])

    # Time Settings
    lookback_days: int = 90  # 최근 3개월

    # Neo4j Settings
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # Vector Settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Paths
    layer3_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def search_queries_path(self) -> Path:
        return self.layer3_dir / "search_queries.yaml"

    @property
    def event_normalization_path(self) -> Path:
        return self.layer3_dir / "event_normalization.yaml"

    def validate(self) -> bool:
        """설정 유효성 검사"""
        if not self.brave_api_key:
            raise ValueError("BRAVE_API_KEY가 설정되지 않았습니다")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
        return True


# Factor 목록 (Layer 2에서 가져옴)
CORE_FACTORS = [
    "물류비", "패널가격", "경쟁심화", "재고정상화", "원재료비",
    "관세", "OLED판매확대", "수요부진", "WebOS매출확대", "해상운임",
    "TV수요부진", "지역별수요", "마케팅비용", "프리미엄제품확대", "환율",
    "소비심리위축", "스포츠이벤트", "글로벌수요", "가전수요", "출하량",
    "판촉비", "비용개선", "비용증가", "원가개선", "성수기효과",
    "수익성", "실적부진", "B2B매출확대", "운송비",
]

# Dimension 목록 (Layer 1에서 가져옴)
DIMENSIONS = {
    "Region": ["NA", "EU", "ASIA", "KR"],
    "ProductCategory": ["OLED", "QNED", "LCD"],
    "Channel": ["RETAIL", "ONLINE"],
}
