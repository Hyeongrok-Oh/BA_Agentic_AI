# LGE HE ERP Knowledge Graph Project

## Project Overview
LG전자 HE(Home Entertainment) 사업부 ERP 데이터를 기반으로 지식그래프를 구축하는 프로젝트

## Data Source
- **Database**: `data/lge_he_erp.db` (SQLite)
- **Domain**: TV 제품 판매 데이터

## Database Schema

### Master Data (MD)
| Table | Description | Rows | Primary Key |
|-------|-------------|------|-------------|
| TBL_MD_PRODUCT | 제품 마스터 | 12 | PRODUCT_ID |

**Columns**: PRODUCT_ID, MODEL_NAME, SERIES, PANEL_TYPE, SCREEN_SIZE, LAUNCH_YEAR, MFG_PLANT

### Organization Data (ORG)
| Table | Description | Rows | Primary Key |
|-------|-------------|------|-------------|
| TBL_ORG_SUBSIDIARY | 법인(지역) 마스터 | 6 | SUBSIDIARY_ID |
| TBL_ORG_CUSTOMER | 고객(거래처) 마스터 | 13 | CUSTOMER_ID |

**TBL_ORG_SUBSIDIARY Columns**: SUBSIDIARY_ID, REGION, CURRENCY
**TBL_ORG_CUSTOMER Columns**: CUSTOMER_ID, CUST_NAME, SUBSIDIARY_ID, CHANNEL_TYPE

### Transaction Data (TX)
| Table | Description | Rows | Primary Key |
|-------|-------------|------|-------------|
| TBL_TX_SALES_HEADER | 판매 주문 헤더 | 10,269 | ORDER_NO |
| TBL_TX_SALES_ITEM | 판매 주문 아이템 | 10,269 | ORDER_NO, ITEM_NO |
| TBL_TX_PRICE_CONDITION | 가격 조건 | 23,797 | ORDER_NO, ITEM_NO, COND_TYPE |
| TBL_TX_COST_DETAIL | 원가 상세 | 41,076 | ORDER_NO, ITEM_NO, COST_TYPE |

## Foreign Key Relationships
```
TBL_ORG_CUSTOMER.SUBSIDIARY_ID -> TBL_ORG_SUBSIDIARY.SUBSIDIARY_ID
TBL_TX_SALES_HEADER.CUSTOMER_ID -> TBL_ORG_CUSTOMER.CUSTOMER_ID
TBL_TX_SALES_ITEM.ORDER_NO -> TBL_TX_SALES_HEADER.ORDER_NO
TBL_TX_SALES_ITEM.PRODUCT_ID -> TBL_MD_PRODUCT.PRODUCT_ID
```

## Sample Data

### Products (OLED TV)
- OLED65G4PUA: 65" OLED evo G4, 2024, Mexico
- OLED77G4PUA: 77" OLED evo G4, 2024, Mexico
- OLED83G4PUA: 83" OLED evo G4, 2024, Poland

### Subsidiaries
- LGEUS: North America (USD)
- LGECA: North America (CAD)
- LGEKR: Korea (KRW)

### Customers
- Best Buy, Costco, Amazon.com (US RETAIL/ONLINE)

## Knowledge Graph 3-Layer Architecture

```
[Layer 3: Event]
    │ INCREASES / DECREASES
    ▼
[Layer 2: Factor]
    │ PROPORTIONAL / INVERSELY_PROPORTIONAL
    ▼
[Layer 1: Anchor (3NF RDB Schema)]
```

### Layer 1: 3NF RDB Anchoring (구현 완료)
- 3정규형 DB의 테이블/컬럼 구조를 그래프로 변환
- Fact 컬럼(매출, 원가 등)을 Anchor 노드로 정의
- 노드: Table, Column, Anchor
- 관계: HAS_COLUMN, HAS_ANCHOR, REFERENCES, BELONGS_TO

### Layer 2: Anchor-Factor 상관관계 (구현 완료)
- 컨센서스 리포트 + LLM으로 외부 Factor 추출
- Factor ─(PROPORTIONAL/INVERSELY_PROPORTIONAL)→ Anchor
- Factor Chain: Factor 간 관계 (예: 유가 → 물류비)
- **464 Factor 노드, 74 AFFECTS, 451 INFLUENCES 관계**

### Layer 3: Event 매핑 (구현 완료)
- Serper API로 뉴스 검색 → LLM으로 Event 추출
- Event ─(INCREASES/DECREASES)→ Factor
- Event ─(TARGETS)→ Dimension (Region, ProductCategory)
- 추론 경로: Event → Factor → Anchor
- **100+ Event 노드, Vector Embedding 포함**

## Anchor Nodes (Fact Columns)
| Table | Column | Metric Type | Description |
|-------|--------|-------------|-------------|
| TBL_TX_SALES_HEADER | TOTAL_NET_VALUE | VALUE | 총 매출액 |
| TBL_TX_SALES_ITEM | NET_VALUE | VALUE | 아이템별 매출 |
| TBL_TX_SALES_ITEM | ORDER_QTY | QTY | 판매 수량 |
| TBL_TX_COST_DETAIL | COST_AMOUNT | VALUE | 원가 |
| TBL_TX_PRICE_CONDITION | COND_VALUE | VALUE | 가격 조건 값 |

## Directory Structure
```
BI/
├── knowledge_graph/         # Layer 1 구현
│   ├── __init__.py
│   ├── config.py           # 설정
│   ├── models.py           # Node/Relationship 타입
│   ├── schema_extractor.py # DB 스키마 추출
│   ├── graph_builder.py    # Neo4j 빌더
│   └── main.py             # 실행 진입점
├── data/
│   └── lge_he_erp.db
├── consensus/              # 컨센서스 리포트 (Layer 2용)
├── dart/                   # DART 공시 (Layer 2용)
├── .env                    # 환경변수
└── PROJECT_CONTEXT.md      # 이 파일
```

## Usage

```bash
# 스키마 추출 (확인용)
python -m knowledge_graph.main extract

# Neo4j에 그래프 빌드
python -m knowledge_graph.main build

# 스키마 JSON 내보내기
python -m knowledge_graph.main export -o schema.json
```

## Neo4j Setup (Docker)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

---

## Multi-Agent BI System Architecture (구현 완료)

### System Overview
```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Intent Classifier             │
│  (LLM-based: service_type, analysis_mode)│
└─────────────────────────────────────────┘
    │
    ├── data_qa + descriptive ──▶ SQL Agent
    │
    └── data_qa + diagnostic ──▶ Analysis Agent (Orchestrator)
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                  │                  │
                   ▼                  ▼                  ▼
           Hypothesis          Hypothesis          Event
           Generator           Validator           Matcher
         (Graph-Enhanced)       (SQL Agent)    (Hybrid Scoring)
```

### Intent Classifier (`intent_classifier/`)
- **서비스 유형**: `data_qa`, `report_generation`
- **분석 모드**: `descriptive` (데이터 조회), `diagnostic` (원인 분석)
- **Sub Intent**: `internal_data`, `external_data`, `event_query`
- **Entity 추출**: period, region, company, metric_type

---

## Analysis Agent Components

### 1. Hypothesis Generator (Graph-Enhanced)
**파일**: `agents/analysis/hypothesis_generator.py`

#### 핵심 기능
- Knowledge Graph에서 KPI 관련 Factor 조회
- 최근 Event 정보 조회
- LLM + Graph Context 기반 가설 생성

#### KPI 매핑
```python
KPI_MAPPING = {
    "매출": {"anchor_id": "revenue", "keywords": ["매출", "revenue", "수익", "sales"]},
    "원가": {"anchor_id": "cost", "keywords": ["원가", "cost", "비용", "expense"]},
    "판매수량": {"anchor_id": "quantity", "keywords": ["판매량", "수량", "quantity", "volume"]},
}
```

#### Graph 조회 쿼리 (Factor)
```cypher
MATCH (f:Factor)-[r:AFFECTS]->(a:Anchor {id: $anchor_id})
RETURN f.name as factor, f.id as factor_id,
       r.type as relation_type, r.mention_count as mention_count
ORDER BY r.mention_count DESC
LIMIT 15
```

#### Graph 조회 쿼리 (Recent Events)
```cypher
MATCH (e:Event)-[r1:INCREASES|DECREASES]->(f:Factor)-[r2:AFFECTS]->(a:Anchor {id: $anchor_id})
OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
RETURN e.name as event_name, e.category, e.severity,
       type(r1) as impact_on_factor, f.name as factor
ORDER BY CASE e.severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 ELSE 3 END
LIMIT 10
```

#### 출력: Hypothesis 데이터 클래스
```python
@dataclass
class Hypothesis:
    id: str
    category: str       # revenue, cost, pricing, external
    factor: str         # 관련 Factor
    direction: str      # increase, decrease
    description: str    # 가설 설명
    sql_template: str   # 검증용 SQL 힌트
    validated: bool
    validation_data: Dict
    graph_evidence: Dict  # from_graph, relation_type, mention_count, related_events
```

---

### 2. Hypothesis Validator (SQL Agent)
**파일**: `agents/analysis/hypothesis_validator.py`

#### 핵심 기능
- 각 가설에 대한 SQL 쿼리 생성 (LLM)
- ERP 데이터베이스에서 실행
- 기간 비교 (전분기 vs 당분기)
- threshold (5%) 초과 시 검증 통과

#### SQL Generation Prompt
```python
SQL_GENERATION_PROMPT = """
ERP 데이터베이스에서 {hypothesis}를 검증하는 SQL을 작성하세요.

원가 유형 (COST_TYPE): MAT(재료비), LOG(물류비), TAR(관세), OH(오버헤드)
가격 조건 (COND_TYPE): ZPR0(매출), ZPRO(PP), K007(할인), ZMDF(MDF)
"""
```

#### 출력: ValidationResult
```python
@dataclass
class ValidationResult:
    hypothesis_id: str
    validated: bool
    change_percent: float
    previous_value: float
    current_value: float
    direction: str
    details: str
    sql_query: str  # 사용된 SQL 쿼리
```

---

### 3. Event Matcher (Hybrid Scoring)
**파일**: `agents/analysis/event_matcher.py`

#### 핵심 기능
- 검증된 가설에 대해 관련 Event 매칭
- **Hybrid Scoring**: Vector Similarity + Knowledge Graph
- 최종 스코어 0-1 범위

#### Scoring Algorithm

##### Overall Weights
```python
WEIGHTS = {
    "semantic": 0.4,  # Vector Similarity (40%)
    "graph": 0.6,     # Graph Score (60%)
}
```

##### Graph Score Components
```python
GRAPH_WEIGHTS = {
    "direction": 0.4,   # 방향 일치 (40%)
    "magnitude": 0.3,   # 크기 (30%)
    "region": 0.2,      # 지역 일치 (20%)
    "severity": 0.1,    # 심각도 (10%)
}

MAGNITUDE_SCORES = {"high": 1.0, "medium": 0.6, "low": 0.3}
SEVERITY_SCORES = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
```

##### Direction Score Logic
```python
def _calc_direction_score(hypothesis, event):
    factor_anchor = event.get("factor_anchor_relation")  # PROPORTIONAL or INVERSELY_PROPORTIONAL
    event_impact = event.get("impact_type")               # INCREASES or DECREASES
    hypothesis_dir = hypothesis.direction                 # increase or decrease

    if factor_anchor == "PROPORTIONAL":
        expected = "INCREASES" if hypothesis_dir == "increase" else "DECREASES"
    else:  # INVERSELY_PROPORTIONAL
        expected = "DECREASES" if hypothesis_dir == "increase" else "INCREASES"

    return 1.0 if event_impact == expected else 0.5
```

##### Vector Search Query (Neo4j)
```cypher
MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor)
WHERE f.name CONTAINS $factor_keyword
WITH e, r, f,
     vector.similarity.cosine(e.embedding, $query_embedding) AS similarity
WHERE similarity > 0.5
RETURN e.id, e.name, e.category, e.severity,
       type(r) as impact_type, r.magnitude,
       f.name as matched_factor,
       similarity as vector_score
ORDER BY similarity DESC
LIMIT $top_k
```

#### 출력: MatchedEvent
```python
@dataclass
class MatchedEvent:
    event_id: str
    event_name: str
    event_category: str
    matched_factor: str
    impact_type: str        # INCREASES or DECREASES
    magnitude: str          # high, medium, low
    severity: str           # critical, high, medium, low
    target_regions: List[str]
    total_score: float      # 0-1 범위
    score_breakdown: Dict   # semantic, graph, direction, magnitude, region, severity
    sources: List[Dict]
    evidence: str
```

---

## Analysis Flow (전체 플로우)

```
Step 1: Intent Classification
    │
    ▼ (diagnostic mode)
Step 2: Graph-Enhanced Hypothesis Generation
    │   - KPI 추출 (질문에서)
    │   - Graph에서 관련 Factor 조회 (mention_count 기준)
    │   - Graph에서 최근 Event 조회
    │   - LLM으로 가설 생성 (Graph context 포함)
    │
    ▼
Step 3: Hypothesis Validation (SQL Agent)
    │   - 각 가설에 대한 SQL 생성
    │   - ERP DB 실행
    │   - 변화율 5% 이상 시 검증 통과
    │   - SQL 쿼리 저장 및 표시
    │
    ▼
Step 4: Event Matching (Hybrid Scoring)
    │   - OpenAI Embedding 생성
    │   - Neo4j Vector Search
    │   - Graph-based Scoring 계산
    │   - Hybrid Score = 0.4*Semantic + 0.6*Graph
    │
    ▼
Step 5: Summary Generation
    │   - 상세 분석 결과 종합
    │   - LLM으로 자연어 요약 생성
    │
    ▼
Final Output
```

---

## Updated Directory Structure

```
BI/
├── agents/                      # Multi-Agent System
│   ├── __init__.py
│   ├── base.py                  # BaseAgent, AgentContext
│   ├── orchestrator.py          # 메인 오케스트레이터
│   ├── search_agent.py          # 검색 에이전트
│   ├── tools/                   # 공용 도구
│   │   ├── __init__.py
│   │   ├── sql_executor.py      # SQL 실행기
│   │   └── graph_executor.py    # Neo4j 실행기
│   └── analysis/                # 분석 에이전트 그룹
│       ├── __init__.py
│       ├── analysis_agent.py    # 분석 오케스트레이터
│       ├── hypothesis_generator.py  # Graph-Enhanced 가설 생성
│       ├── hypothesis_validator.py  # SQL 기반 가설 검증
│       ├── event_matcher.py     # Hybrid Scoring 이벤트 매칭
│       └── evidence_collector.py    # (deprecated)
│
├── knowledge_graph/             # Knowledge Graph 구축
│   ├── layer1/                  # 3NF RDB Anchoring
│   ├── layer2/                  # Factor-Anchor 관계
│   │   ├── models.py
│   │   ├── factor_extractor.py  # LLM 기반 Factor 추출
│   │   ├── normalizer.py        # Factor 정규화
│   │   └── neo4j_loader.py
│   └── layer3/                  # Event Layer
│       ├── models.py
│       ├── serper_client.py     # 뉴스 검색 API
│       ├── event_extractor.py   # LLM 기반 Event 추출
│       ├── factor_linker.py     # Event → Factor 매핑
│       ├── dimension_linker.py  # Event → Dimension 매핑
│       ├── vector_store.py      # Embedding 생성
│       └── neo4j_loader.py
│
├── intent_classifier/           # Intent 분류기
│   └── src/
│       └── intent_classifier/
│           ├── classifier.py
│           └── prompts.py
│
├── sql/                         # SQL Agent 관련
├── data/
│   └── lge_he_erp.db           # ERP 데이터베이스
│
├── app.py                       # Streamlit UI
├── .env                         # 환경변수
└── PROJECT_CONTEXT.md           # 이 파일
```

---

## Environment Variables (.env)

```bash
OPENAI_API_KEY=sk-xxx
SERPER_API_KEY=xxx
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
DB_PATH=/Users/hyeongrokoh/BI/data/lge_he_erp.db
```

---

## Neo4j Graph Statistics

### Node Counts
| Label | Count | Description |
|-------|-------|-------------|
| Anchor | 3 | 매출, 원가, 판매수량 |
| Factor | 464 | 외부 영향 요인 |
| Event | 100+ | 뉴스 기반 이벤트 |
| Dimension | 10+ | Region, ProductCategory |

### Relationship Counts
| Type | Count | Description |
|------|-------|-------------|
| AFFECTS | 74 | Factor → Anchor |
| INFLUENCES | 451 | Factor → Factor |
| INCREASES | 150+ | Event → Factor |
| DECREASES | 50+ | Event → Factor |
| TARGETS | 100+ | Event → Dimension |

### Vector Index
```cypher
CREATE VECTOR INDEX event_embedding IF NOT EXISTS
FOR (e:Event) ON e.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
```

---

## Sample Queries

### 가설 생성 시 Factor 조회
```cypher
MATCH (f:Factor)-[r:AFFECTS]->(a:Anchor {id: "cost"})
RETURN f.name, r.type, r.mention_count
ORDER BY r.mention_count DESC LIMIT 10
```

### 하이브리드 이벤트 검색
```cypher
MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor {name: "물류비"})
OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
WITH e, r, f, collect(d.name) as regions,
     vector.similarity.cosine(e.embedding, $query_embedding) AS similarity
WHERE similarity > 0.5
RETURN e.name, type(r), r.magnitude, e.severity, regions, similarity
ORDER BY similarity DESC
LIMIT 5
```

### 인과관계 추론 경로
```cypher
MATCH path = (e:Event)-[:INCREASES]->(f:Factor)-[:AFFECTS]->(a:Anchor {id: "cost"})
WHERE e.severity IN ['critical', 'high']
RETURN e.name as event, f.name as factor, a.name as anchor
```

---

## Reasoning-based Answer Generation (구현 완료)

### 추론 프롬프트 (REASONING_PROMPT)
```python
REASONING_PROMPT = """
## 질문
{question}

## 내부 데이터 분석 결과 (ERP 기반)
{internal_data}

## 외부 이벤트 정보 (뉴스 기반)
{external_events}

## 추론 태스크
1. **인과관계 추론**: Event → Factor → KPI 경로
2. **영향도 분석**: 변화율/이벤트 심각도 기준
3. **종합 결론**: 핵심 원인 2-3가지

## 응답 형식
### 분석 결론
[핵심 원인 요약]

### 상세 분석
**1. [원인1] (변화율: +XX%)**
- 내부 데이터: [수치 변화]
- 외부 요인: [이벤트명]
- 인과 경로: [Event] → [Factor] → [KPI 영향]
- 출처: [1]

### 출처
[1] 기사제목 - URL
"""
```

### 출처 수집 로직
```python
def _generate_summary(self, question: str, details: List[Dict]) -> Dict:
    # 1. 내부 데이터 포맷팅 (ERP)
    # 2. 외부 이벤트 + 출처 수집 (Neo4j source_urls, source_titles)
    # 3. 추론 프롬프트 구성
    # 4. gpt-4o 호출 (temperature=0.2)
    # 5. 출처 섹션 추가

    return {
        "summary": summary,  # 추론 결과
        "sources": all_sources  # 출처 목록
    }
```

### 출처 데이터 구조
```python
source = {
    "idx": 1,
    "title": "European Commission seeks relief from Section 232 tariffs",
    "url": "https://www.steelmarketupdate.com/...",
    "event": "미국 행정부 섹션 232 관세 확대",
    "factor": "물류비"
}
```

---

## Test Results (2024-12-18)

### 추론 기반 분석 테스트 (최신)
```
질문: "2024년 4분기 북미 지역의 원가가 증가한 원인을 분석해줘"

결과:
- 가설 생성: 6개 (Graph-Enhanced)
- 가설 검증: 2개 통과 (물류비 +206.3%, 경쟁심화 +33.5%)
- 이벤트 매칭: 10개 (Score 0.78-0.80)
- 출처 수집: 3개 URL

추론 결과 (요약):
### 분석 결론
2024년 4분기 북미 지역 원가 증가의 주요 원인은 미국 행정부의 섹션 232
관세 확대와 수출입 운송비용 상승입니다.

### 상세 분석
**1. 물류비 증가 (변화율: +206.3%)**
- 인과 경로: [섹션 232 관세 확대] → [수입 철강/알루미늄 비용 증가] → [물류비]
- 출처: [1]

**2. 경쟁심화 (변화율: +33.5%)**
- 인과 경로: [원/달러 환율 상승] → [가격 경쟁력 약화] → [경쟁심화]
- 출처: [3]

### 출처
[1] European Commission seeks relief from Section 232 tariffs
    https://www.steelmarketupdate.com/...
[2] Trump's steel and aluminum tariffs are putting America first
    https://thehill.com/...
[3] 관세청. 2025년 11월 수출입 운송비용 현황 발표
    https://policenews24.co.kr/...
```

---

## Future Enhancements

1. **Real-time Event Ingestion**: 뉴스 API 연동으로 실시간 Event 추가
2. **Confidence Calibration**: Hybrid Score 가중치 최적화
3. **Multi-hop Reasoning**: Event → Factor → Factor → Anchor 체인 추론
4. **Report Generation**: 분석 결과 자동 리포트 생성
5. **Feedback Loop**: 사용자 피드백으로 모델 개선
