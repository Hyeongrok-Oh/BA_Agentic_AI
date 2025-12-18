"""
Neo4j 초기화 스크립트
Knowledge Graph Layer 1, 2, 3 데이터를 Neo4j에 적재합니다.
"""

import os
import sys
import time

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase


def wait_for_neo4j(uri: str, user: str, password: str, max_retries: int = 30):
    """Neo4j가 준비될 때까지 대기"""
    print("Waiting for Neo4j to be ready...")
    for i in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            driver.close()
            print("Neo4j is ready!")
            return True
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries}: Neo4j not ready yet... ({e})")
            time.sleep(2)
    return False


def init_constraints(driver):
    """제약조건 및 인덱스 생성"""
    constraints = [
        # Layer 1
        "CREATE CONSTRAINT anchor_id IF NOT EXISTS FOR (a:Anchor) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (r:Region) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:ProductCategory) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT channel_id IF NOT EXISTS FOR (c:Channel) REQUIRE c.id IS UNIQUE",
        # Layer 2
        "CREATE CONSTRAINT factor_id IF NOT EXISTS FOR (f:Factor) REQUIRE f.id IS UNIQUE",
        # Layer 3
        "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
    ]

    indexes = [
        "CREATE INDEX factor_category IF NOT EXISTS FOR (f:Factor) ON (f.category)",
        "CREATE INDEX event_category IF NOT EXISTS FOR (e:Event) ON (e.category)",
        "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.start_date)",
    ]

    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"Created: {constraint[:60]}...")
            except Exception as e:
                print(f"Skip (exists): {constraint[:40]}...")

        for index in indexes:
            try:
                session.run(index)
                print(f"Created: {index[:60]}...")
            except Exception as e:
                print(f"Skip (exists): {index[:40]}...")


def init_layer1(driver):
    """Layer 1: Anchor 및 Dimension 노드 생성"""
    print("\n=== Initializing Layer 1 (Anchor & Dimension) ===")

    # Anchor 노드
    anchors = [
        {"id": "revenue", "name": "매출", "name_en": "Revenue", "unit": "억원"},
        {"id": "cost", "name": "원가", "name_en": "Cost", "unit": "억원"},
        {"id": "sales_volume", "name": "판매수량", "name_en": "Sales Volume", "unit": "만대"},
    ]

    # Region Dimension
    regions = [
        {"id": "NA", "name": "북미", "name_en": "North America"},
        {"id": "EU", "name": "유럽", "name_en": "Europe"},
        {"id": "ASIA", "name": "아시아", "name_en": "Asia"},
        {"id": "KR", "name": "한국", "name_en": "Korea"},
        {"id": "LATAM", "name": "중남미", "name_en": "Latin America"},
        {"id": "MEA", "name": "중동/아프리카", "name_en": "Middle East & Africa"},
    ]

    # Product Category Dimension
    products = [
        {"id": "OLED", "name": "OLED TV", "is_premium": True},
        {"id": "QNED", "name": "QNED TV", "is_premium": True},
        {"id": "LCD", "name": "LCD TV", "is_premium": False},
        {"id": "LIFESTYLE", "name": "라이프스타일 TV", "is_premium": True},
    ]

    # Channel Dimension
    channels = [
        {"id": "RETAIL", "name": "소매", "name_en": "Retail"},
        {"id": "ONLINE", "name": "온라인", "name_en": "Online"},
        {"id": "B2B", "name": "B2B", "name_en": "Business to Business"},
    ]

    with driver.session() as session:
        # Create Anchors
        for anchor in anchors:
            session.run("""
                MERGE (a:Anchor {id: $id})
                SET a.name = $name, a.name_en = $name_en, a.unit = $unit, a.layer = 1
            """, anchor)
        print(f"Created {len(anchors)} Anchor nodes")

        # Create Regions
        for region in regions:
            session.run("""
                MERGE (r:Region:Dimension {id: $id})
                SET r.name = $name, r.name_en = $name_en, r.layer = 1
            """, region)
        print(f"Created {len(regions)} Region nodes")

        # Create Products
        for product in products:
            session.run("""
                MERGE (p:ProductCategory:Dimension {id: $id})
                SET p.name = $name, p.is_premium = $is_premium, p.layer = 1
            """, product)
        print(f"Created {len(products)} ProductCategory nodes")

        # Create Channels
        for channel in channels:
            session.run("""
                MERGE (c:Channel:Dimension {id: $id})
                SET c.name = $name, c.name_en = $name_en, c.layer = 1
            """, channel)
        print(f"Created {len(channels)} Channel nodes")


def init_layer2_sample(driver):
    """Layer 2: 핵심 Factor 노드 및 관계 생성 (샘플)"""
    print("\n=== Initializing Layer 2 (Factor) - Sample ===")

    # 핵심 Factor 노드
    factors = [
        # 가격/마케팅
        {"id": "price_competitiveness", "name": "가격경쟁력", "category": "pricing", "direction": "PROPORTIONAL"},
        {"id": "marketing_expense", "name": "마케팅비", "category": "marketing", "direction": "PROPORTIONAL"},
        {"id": "promotion_intensity", "name": "프로모션 강도", "category": "marketing", "direction": "PROPORTIONAL"},
        # 원가
        {"id": "panel_price", "name": "패널가격", "category": "component", "direction": "INVERSE"},
        {"id": "logistics_cost", "name": "물류비", "category": "logistics", "direction": "INVERSE"},
        {"id": "shipping_cost", "name": "해상운임", "category": "logistics", "direction": "INVERSE"},
        {"id": "raw_material_cost", "name": "원재료비", "category": "component", "direction": "INVERSE"},
        {"id": "tariff", "name": "관세", "category": "trade", "direction": "INVERSE"},
        {"id": "exchange_rate", "name": "환율", "category": "macro", "direction": "COMPLEX"},
        # 수요
        {"id": "consumer_demand", "name": "소비수요", "category": "demand", "direction": "PROPORTIONAL"},
        {"id": "housing_market", "name": "주택경기", "category": "macro", "direction": "PROPORTIONAL"},
        {"id": "consumer_confidence", "name": "소비심리", "category": "macro", "direction": "PROPORTIONAL"},
        # 경쟁
        {"id": "competition_intensity", "name": "경쟁심화", "category": "competition", "direction": "INVERSE"},
        {"id": "market_share", "name": "시장점유율", "category": "competition", "direction": "PROPORTIONAL"},
        # 제품
        {"id": "product_lineup", "name": "제품라인업", "category": "product", "direction": "PROPORTIONAL"},
        {"id": "technology_advantage", "name": "기술우위", "category": "product", "direction": "PROPORTIONAL"},
    ]

    with driver.session() as session:
        for factor in factors:
            session.run("""
                MERGE (f:Factor {id: $id})
                SET f.name = $name, f.category = $category,
                    f.direction = $direction, f.layer = 2
            """, factor)
        print(f"Created {len(factors)} Factor nodes")

        # AFFECTS 관계 (Factor → Anchor)
        affects_relations = [
            # Revenue
            ("price_competitiveness", "revenue", "PROPORTIONAL"),
            ("marketing_expense", "revenue", "PROPORTIONAL"),
            ("consumer_demand", "revenue", "PROPORTIONAL"),
            ("market_share", "revenue", "PROPORTIONAL"),
            ("product_lineup", "revenue", "PROPORTIONAL"),
            # Cost
            ("panel_price", "cost", "PROPORTIONAL"),
            ("logistics_cost", "cost", "PROPORTIONAL"),
            ("shipping_cost", "cost", "PROPORTIONAL"),
            ("raw_material_cost", "cost", "PROPORTIONAL"),
            ("tariff", "cost", "PROPORTIONAL"),
            ("exchange_rate", "cost", "COMPLEX"),
            # Sales Volume
            ("consumer_demand", "sales_volume", "PROPORTIONAL"),
            ("price_competitiveness", "sales_volume", "PROPORTIONAL"),
            ("promotion_intensity", "sales_volume", "PROPORTIONAL"),
        ]

        for factor_id, anchor_id, direction in affects_relations:
            session.run("""
                MATCH (f:Factor {id: $factor_id})
                MATCH (a:Anchor {id: $anchor_id})
                MERGE (f)-[r:AFFECTS]->(a)
                SET r.direction = $direction
            """, {"factor_id": factor_id, "anchor_id": anchor_id, "direction": direction})
        print(f"Created {len(affects_relations)} AFFECTS relations")

        # INFLUENCES 관계 (Factor → Factor)
        influences_relations = [
            ("shipping_cost", "logistics_cost", "PROPORTIONAL"),
            ("exchange_rate", "raw_material_cost", "PROPORTIONAL"),
            ("housing_market", "consumer_demand", "PROPORTIONAL"),
            ("consumer_confidence", "consumer_demand", "PROPORTIONAL"),
            ("competition_intensity", "price_competitiveness", "INVERSE"),
            ("technology_advantage", "market_share", "PROPORTIONAL"),
        ]

        for from_id, to_id, direction in influences_relations:
            session.run("""
                MATCH (f1:Factor {id: $from_id})
                MATCH (f2:Factor {id: $to_id})
                MERGE (f1)-[r:INFLUENCES]->(f2)
                SET r.direction = $direction
            """, {"from_id": from_id, "to_id": to_id, "direction": direction})
        print(f"Created {len(influences_relations)} INFLUENCES relations")


def init_layer3_sample(driver):
    """Layer 3: 핵심 Event 노드 및 관계 생성 (샘플)"""
    print("\n=== Initializing Layer 3 (Event) - Sample ===")

    events = [
        {
            "id": "red_sea_crisis_2024",
            "name": "홍해 사태",
            "name_en": "Red Sea Crisis",
            "category": "geopolitical",
            "start_date": "2023-11-19",
            "is_ongoing": True,
            "severity": "high",
            "evidence": "후티 반군의 홍해 선박 공격으로 국제 해상운송 비용 급등"
        },
        {
            "id": "trump_tariff_2025",
            "name": "트럼프 관세 정책",
            "name_en": "Trump Tariff Policy",
            "category": "policy",
            "start_date": "2025-01-20",
            "is_ongoing": True,
            "severity": "high",
            "evidence": "미국 수입 제품에 대한 25% 관세 부과 발표"
        },
        {
            "id": "lcd_oversupply_2024",
            "name": "LCD 패널 공급과잉",
            "name_en": "LCD Panel Oversupply",
            "category": "market",
            "start_date": "2024-06-01",
            "is_ongoing": True,
            "severity": "medium",
            "evidence": "중국 BOE, CSOT 등 패널 업체들의 생산량 증가로 공급 과잉"
        },
        {
            "id": "usd_krw_surge_2024",
            "name": "원/달러 환율 급등",
            "name_en": "USD/KRW Exchange Rate Surge",
            "category": "macro_economy",
            "start_date": "2024-10-01",
            "is_ongoing": True,
            "severity": "high",
            "evidence": "달러 강세로 원/달러 환율 1,400원 돌파"
        },
        {
            "id": "black_friday_2024",
            "name": "블랙프라이데이 2024",
            "name_en": "Black Friday 2024",
            "category": "market",
            "start_date": "2024-11-29",
            "is_ongoing": False,
            "severity": "medium",
            "evidence": "미국 블랙프라이데이 시즌 TV 판매 호조"
        },
    ]

    with driver.session() as session:
        for event in events:
            session.run("""
                MERGE (e:Event {id: $id})
                SET e.name = $name, e.name_en = $name_en, e.category = $category,
                    e.start_date = date($start_date), e.is_ongoing = $is_ongoing,
                    e.severity = $severity, e.evidence = $evidence, e.layer = 3
            """, event)
        print(f"Created {len(events)} Event nodes")

        # Event → Factor 관계
        event_factor_relations = [
            # 홍해 사태
            ("red_sea_crisis_2024", "shipping_cost", "INCREASES", "high"),
            ("red_sea_crisis_2024", "logistics_cost", "INCREASES", "high"),
            # 트럼프 관세
            ("trump_tariff_2025", "tariff", "INCREASES", "high"),
            # LCD 공급과잉
            ("lcd_oversupply_2024", "panel_price", "DECREASES", "medium"),
            # 환율 급등
            ("usd_krw_surge_2024", "exchange_rate", "INCREASES", "high"),
            ("usd_krw_surge_2024", "raw_material_cost", "INCREASES", "medium"),
            # 블랙프라이데이
            ("black_friday_2024", "consumer_demand", "INCREASES", "medium"),
        ]

        for event_id, factor_id, rel_type, magnitude in event_factor_relations:
            session.run(f"""
                MATCH (e:Event {{id: $event_id}})
                MATCH (f:Factor {{id: $factor_id}})
                MERGE (e)-[r:{rel_type}]->(f)
                SET r.magnitude = $magnitude
            """, {"event_id": event_id, "factor_id": factor_id, "magnitude": magnitude})
        print(f"Created {len(event_factor_relations)} Event-Factor relations")

        # Event → Dimension 관계 (TARGETS)
        event_dimension_relations = [
            ("red_sea_crisis_2024", "EU", "Region"),
            ("red_sea_crisis_2024", "ASIA", "Region"),
            ("trump_tariff_2025", "NA", "Region"),
            ("lcd_oversupply_2024", "LCD", "ProductCategory"),
            ("black_friday_2024", "NA", "Region"),
        ]

        for event_id, dim_id, dim_type in event_dimension_relations:
            session.run(f"""
                MATCH (e:Event {{id: $event_id}})
                MATCH (d:{dim_type} {{id: $dim_id}})
                MERGE (e)-[r:TARGETS]->(d)
            """, {"event_id": event_id, "dim_id": dim_id})
        print(f"Created {len(event_dimension_relations)} TARGETS relations")


def main():
    # 환경변수에서 설정 읽기
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")

    print(f"Connecting to Neo4j at {uri}")

    # Neo4j 대기
    if not wait_for_neo4j(uri, user, password):
        print("Failed to connect to Neo4j")
        sys.exit(1)

    # 드라이버 생성
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        # 초기화 실행
        init_constraints(driver)
        init_layer1(driver)
        init_layer2_sample(driver)
        init_layer3_sample(driver)

        # 결과 확인
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            print("\n=== Summary ===")
            for record in result:
                print(f"  {record['label']}: {record['count']} nodes")

            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
            """)
            print("\nRelationships:")
            for record in rel_result:
                print(f"  {record['rel_type']}: {record['count']} relations")

        print("\n✅ Neo4j initialization complete!")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
