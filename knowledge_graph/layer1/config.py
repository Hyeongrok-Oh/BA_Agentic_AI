"""Layer 1 전용 설정 - Dimension, Anchor 정의"""

from dataclasses import dataclass, field
from typing import List, Dict

from ..config import BaseConfig


@dataclass
class Layer1Config(BaseConfig):
    """Layer 1: Dimension & Anchor 설정"""

    # Anchor definitions (KPI 중심점)
    anchors: List[Dict] = field(default_factory=lambda: [
        {
            "id": "revenue",
            "metric_type": "REVENUE",
            "name": "매출",
            "description": "총 매출액 (Net Value)",
            "source_table": "TBL_TX_SALES_ITEM",
            "source_column": "NET_VALUE",
        },
        {
            "id": "quantity",
            "metric_type": "QUANTITY",
            "name": "판매수량",
            "description": "판매 수량",
            "source_table": "TBL_TX_SALES_ITEM",
            "source_column": "ORDER_QTY",
        },
        {
            "id": "cost",
            "metric_type": "COST",
            "name": "원가",
            "description": "제품 원가",
            "source_table": "TBL_TX_COST_DETAIL",
            "source_column": "COST_AMOUNT",
        },
    ])

    # 상위 레벨 Dimension 정의
    product_categories: List[Dict] = field(default_factory=lambda: [
        {"id": "cat_oled", "name": "OLED", "description": "OLED TV"},
        {"id": "cat_qned", "name": "QNED", "description": "QNED Mini LED TV"},
        {"id": "cat_lcd", "name": "LCD", "description": "LCD/LED TV"},
    ])

    regions: List[Dict] = field(default_factory=lambda: [
        {"id": "region_na", "name": "NA", "description": "North America"},
        {"id": "region_eu", "name": "EU", "description": "Europe"},
        {"id": "region_kr", "name": "KR", "description": "Korea"},
        {"id": "region_asia", "name": "ASIA", "description": "Asia Pacific"},
    ])

    channels: List[Dict] = field(default_factory=lambda: [
        {"id": "channel_retail", "name": "RETAIL", "description": "오프라인 리테일"},
        {"id": "channel_online", "name": "ONLINE", "description": "온라인 채널"},
    ])

    # Product → Category 매핑 규칙
    product_category_rules: Dict[str, str] = field(default_factory=lambda: {
        "OLED": "cat_oled",
        "QNED": "cat_qned",
    })

    # Subsidiary → Region 매핑
    subsidiary_region_map: Dict[str, str] = field(default_factory=lambda: {
        "LGEUS": "region_na",
        "LGECA": "region_na",
        "LGEKR": "region_kr",
        "LGEUK": "region_eu",
        "LGEDE": "region_eu",
        "LGEJP": "region_asia",
    })


# Alias for backward compatibility
Config = Layer1Config
