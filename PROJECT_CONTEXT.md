# LGE HE BI System - Knowledge Graph + Multi-Agent 기반 KPI 원인 분석

## 1. 프로젝트 개요

### 목적
LG전자 HE(Home Entertainment) 사업부의 ERP 데이터와 외부 뉴스/컨센서스 리포트를 결합하여, KPI(매출/원가/수량) 변동의 **근본 원인을 자동 분석**하는 시스템

### 핵심 기능
1. **자연어 질문 → 원인 분석**: "2024년 4분기 북미 매출이 감소한 원인은?"
2. **Knowledge Graph 기반 인과관계 추론**: Event → Factor → KPI
3. **ERP 데이터 기반 정량 검증**: SQL 쿼리로 가설 검증
4. **Shapley Value 기반 기여도 분석**: 각 Driver가 KPI 변화에 기여한 정도 산출

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         사용자 질문 (Streamlit UI)                           │
│  "2024년 4분기 북미 지역의 매출이 감소한 원인을 분석해줘"                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Intent Classifier                                   │
│    service_type: data_qa | report_generation                                │
│    analysis_mode: descriptive | diagnostic                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
    ┌─────────────────┐                         ┌─────────────────────────┐
    │   SQL Agent     │                         │    Analysis Agent       │
    │ (descriptive)   │                         │    (diagnostic)         │
    └─────────────────┘                         └─────────────────────────┘
                                                            │
                                ┌───────────────────────────┼───────────────────────────┐
                                │                           │                           │
                                ▼                           ▼                           ▼
                    ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
                    │   Hypothesis      │     │   Hypothesis      │     │    Event          │
                    │   Generator       │     │   Validator       │     │    Matcher        │
                    │ (Graph-Enhanced)  │     │ (Tier + Shapley)  │     │ (Hybrid Scoring)  │
                    └───────────────────┘     └───────────────────┘     └───────────────────┘
                                │                       │                       │
                                ▼                       ▼                       ▼
                    ┌───────────────────────────────────────────────────────────────────────┐
                    │                      Knowledge Graph (Neo4j)                          │
                    │  Layer 3: Event → Layer 2: Factor → Layer 1: Anchor (ERP Schema)     │
                    └───────────────────────────────────────────────────────────────────────┘
```

---

## 3. Knowledge Graph 3-Layer 아키텍처

```
[Layer 3: Event]           "트럼프 관세 정책 발표" (2024-11-15)
    │ AFFECTS (polarity: -1, weight: 0.8)
    ▼
[Layer 2: Factor/Driver]   "물류비", "패널원가", "환율"
    │ HYPOTHESIZED_TO_AFFECT (polarity: "+", confidence: 0.85)
    ▼
[Layer 1: Anchor (KPI)]    "매출(revenue)", "원가(cost)", "수량(quantity)"
    │ Mapped to ERP Tables
    ▼
[ERP Database]             TR_SALES, TR_PURCHASE, TR_EXPENSE
```

### Layer 1: 3NF RDB Anchoring
- ERP 테이블/컬럼 구조를 그래프로 변환
- **Anchor 노드**: 매출(revenue), 원가(cost), 판매수량(quantity)
- 노드: `Table`, `Column`, `Anchor`
- 관계: `HAS_COLUMN`, `HAS_ANCHOR`, `REFERENCES`

### Layer 2: Factor-Anchor 관계
- 컨센서스 리포트 + LLM으로 Factor(Driver) 추출
- **464 Factor 노드, 74 AFFECTS, 451 INFLUENCES 관계**
- 관계 유형:
  - `AFFECTS`: Factor → Anchor (직접 영향)
  - `INFLUENCES`: Factor → Factor (간접 영향)

### Layer 3: Event 매핑
- Serper API + LLM으로 뉴스 기반 Event 추출
- **100+ Event 노드, Vector Embedding (1536차원)**
- 관계:
  - `AFFECTS`: Event → Driver (polarity: +1/-1, weight: 0-1)
  - `TARGETS`: Event → Dimension (Region, TimePeriod)

---

## 4. Multi-Agent System

### 4.1 Analysis Agent (Orchestrator)
**파일**: `agents/analysis/analysis_agent.py`

분석 플로우 조율:
```
Step 0: KPI 변동 계산 (TR_SALES 쿼리)
    ↓
Step 1: 가설 생성 (Graph-Enhanced)
    ↓
Step 2: 가설 검증 (Tier 기반 + Shapley)
    ↓
Step 3: 이벤트 매칭 (Hybrid Scoring)
    ↓
Step 4: 결과 종합 + 추론 기반 요약
```

### 4.2 Hypothesis Generator
**파일**: `agents/analysis/hypothesis_generator.py`

**기능**:
- Knowledge Graph에서 KPI 관련 Factor 조회
- 최근 Event 정보 기반 가설 생성
- LLM + Graph Context 활용

**KPI 매핑**:
```python
KPI_MAPPING = {
    "매출": {"anchor_id": "revenue", "keywords": ["매출", "revenue", "sales"]},
    "원가": {"anchor_id": "cost", "keywords": ["원가", "cost", "비용"]},
    "판매수량": {"anchor_id": "quantity", "keywords": ["판매량", "수량"]}
}
```

**Hypothesis 데이터 클래스**:
```python
@dataclass
class Hypothesis:
    id: str
    category: str           # revenue, cost, pricing, external
    factor: str             # 관련 Factor (한글)
    driver_id: str          # Driver ID (한글)
    direction: str          # increase, decrease
    description: str        # 가설 설명
    confidence: float       # 0-1
    consensus_grade: str    # A, B, C
    validated: bool
    validation_data: Dict
    graph_evidence: Dict    # validation_tier, evidence_sentences
```

### 4.3 Hypothesis Validator (V3: Tier + Shapley)
**파일**: `agents/analysis/hypothesis_validator.py`

**검증 방식 3단계 Fallback**:
```
T1 (ERP 직접 검증) → T2 (Proxy 지표) → T3 (Graph 기반)
```

**T1 ERP Drivers (직접 검증 가능)**:
```python
T1_ERP_DRIVERS = {
    "출하량": {"table": "TR_SALES", "column": "QTY"},
    "판매량": {"table": "TR_SALES", "column": "QTY"},
    "OLED비중": {"table": "TR_SALES", "column": "REVENUE_USD", "filter": "prod.DISPLAY_TYPE = 'OLED'"},
    "할인율": {"table": "TR_EXPENSE", "column": "PROMOTION_COST"},
    "패널원가": {"table": "TR_PURCHASE", "column": "PANEL_PRICE_USD"},
    "물류비": {"table": "TR_EXPENSE", "column": "LOGISTICS_COST"},
    "달러환율": {"table": "EXT_MACRO", "column": "EXCHANGE_RATE_KRW_USD"},
    # ... 총 18개
}
```

**T2 Proxy Drivers (외부 지표)**:
```python
T2_PROXY_DRIVERS = {
    "글로벌TV수요": {"table": "EXT_MARKET", "column": "TOTAL_SHIPMENT_10K"},
    "소비심리": {"table": "EXT_MACRO", "column": "CSI_INDEX"},
    "인플레이션": {"table": "EXT_MACRO", "column": "INFLATION_RATE"},
    "금리": {"table": "EXT_MACRO", "column": "INTEREST_RATE"},
    # ... 총 14개
}
```

**T3 Event Drivers (Graph 기반)**:
```python
T3_EVENT_DRIVERS = [
    "경쟁사가격인하", "경쟁사신제품", "공급망차질",
    "무역규제", "스포츠이벤트", "브랜드이슈", ...
]
```

**동적 임계값 계산**:
```python
# 변동계수(CV) 기반 임계값 = stddev / mean × 100 × 1.5
threshold = max(2.0, min(20.0, CV × STDDEV_MULTIPLIER))
```

### 4.4 Event Matcher (Hybrid Scoring)
**파일**: `agents/analysis/event_matcher.py`

**Hybrid Score 공식**:
```
Total Score = 0.4 × Semantic Score + 0.6 × Graph Score

Graph Score = 0.4 × Direction + 0.3 × Magnitude + 0.2 × Region + 0.1 × Severity
```

**MatchedEvent 데이터 클래스**:
```python
@dataclass
class MatchedEvent:
    event_id: str
    event_name: str
    event_category: str
    matched_factor: str
    impact_type: str        # INCREASES or DECREASES
    polarity: int           # +1 or -1
    weight: float           # 0-1
    severity: str           # critical, high, medium, low
    total_score: float      # 0-1 최종 점수
    score_breakdown: Dict
    sources: List[Dict]     # 뉴스 출처
    evidence: str
    start_date: str
    target_regions: List[str]
    target_periods: List[str]
```

---

## 5. Shapley Value 기반 기여도 분석 (신규)

### 5.1 개요
**파일**: `agents/tools/data_analysis.py`

기존 임계값 기반 검증의 한계 극복:
- **기존**: "Driver가 변했는가?" (단순 변동 감지)
- **개선**: "Driver가 KPI 변화에 얼마나 기여했는가?" (기여도 분석)

### 5.2 DataAnalyzer 클래스

```python
class DataAnalyzer(BaseTool):
    """KPI 기여도 분석기 (Shapley Value 기반)"""

    def analyze(
        self,
        hypotheses: List[Hypothesis],
        kpi_id: str = "revenue",
        period: Dict = None,
        months: int = 24
    ) -> AnalysisResult:
        """
        1. 시계열 수집 (Driver별 24개월)
        2. Ridge 회귀 모델 학습
        3. SHAP 기여도 계산
        4. 가설 검증 (rank ≤ 3 OR contribution ≥ 10%)
        5. 해석 생성
        """
```

### 5.3 분석 흐름

```
[Step 1] 시계열 데이터 수집
    └─ Driver별 24개월 월별 데이터 (SQL)
    └─ KPI 24개월 월별 데이터

[Step 2] 회귀모델 학습
    └─ Ridge 회귀 (다중공선성 완화)
    └─ StandardScaler로 표준화
    └─ R² 확인 (모델 설명력)

[Step 3] Shapley Value 계산
    └─ SHAP LinearExplainer
    └─ 각 Driver의 기여도(%) 산출
    └─ 순위 부여

[Step 4] 가설 검증
    └─ rank ≤ 3 → validated
    └─ contribution_pct ≥ 10% → validated
```

### 5.4 Driver-Table 매핑

```python
DRIVER_CONFIG = {
    # T1: ERP 직접 검증
    "출하량": {"table": "TR_SALES", "column": "QTY", "date_col": "SALES_DATE"},
    "TV평균판매가": {"table": "TR_SALES", "column": "REVENUE_USD/QTY", "date_col": "SALES_DATE"},
    "할인율": {"table": "TR_EXPENSE", "column": "PROMOTION_COST", "date_col": "EXPENSE_DATE"},
    "패널원가": {"table": "TR_PURCHASE", "column": "PANEL_PRICE_USD", "date_col": "PURCHASE_DATE"},
    "물류비": {"table": "TR_EXPENSE", "column": "LOGISTICS_COST", "date_col": "EXPENSE_DATE"},
    "달러환율": {"table": "EXT_MACRO", "column": "EXCHANGE_RATE_KRW_USD", "date_col": "DATA_DATE"},
    # T2: Proxy
    "글로벌TV수요": {"table": "EXT_MARKET", "column": "TOTAL_SHIPMENT_10K", "date_col": "DATA_DATE"},
    "소비심리": {"table": "EXT_MACRO", "column": "CSI_INDEX", "date_col": "DATA_DATE"},
}
```

### 5.5 출력 데이터 클래스

```python
@dataclass
class DriverContribution:
    driver_id: str
    driver_name: str
    shapley_value: float
    contribution_pct: float     # 기여도 %
    direction: str              # "positive" | "negative"
    rank: int                   # 기여도 순위
    interpretation: str

@dataclass
class HypothesisValidation:
    hypothesis_id: str
    driver_id: str
    validation_status: str      # "validated" | "not_validated"
    confidence_score: float     # 0-1
    reasoning: str

@dataclass
class AnalysisResult:
    kpi_change_summary: str
    kpi_change_pct: float
    top_drivers: List[DriverContribution]
    hypotheses: List[HypothesisValidation]
    final_explanation: str
    model_r_squared: float
    data_quality: Dict
```

### 5.6 검증 결과 예시

```
매출 감소 -100억 원인 분석:

┌────────────────────────────────────────────────────┐
│  #1 🔴 환율 상승        ████████████████  42.3%    │
│  #2 🔴 할인율 증가      ██████████        28.1%    │
│  #3 🔴 글로벌 수요 감소  ██████           15.2%    │
│  #4 🔴 패널 원가        ███               8.7%     │
│  #5    기타/잔차        ██                5.7%     │
└────────────────────────────────────────────────────┘

→ 할인율: 기여도 28.1% (#2위) → ✅ validated
→ 패널원가: 기여도 8.7% (#4위) → ❌ not_validated
```

### 5.7 HypothesisValidator 통합

```python
# hypothesis_validator.py
class HypothesisValidator:
    def __init__(self, ...):
        self.data_analyzer = DataAnalyzer(self.sql_executor)

    def validate_with_shapley(
        self,
        hypotheses: List[Hypothesis],
        kpi_id: str = "revenue"
    ) -> Dict[str, Any]:
        """Shapley Value 기반 가설 검증"""
        analysis_result = self.data_analyzer.analyze(hypotheses, kpi_id)

        return {
            "analysis_result": analysis_result,
            "validated_hypotheses": [...],
            "contributions": [...],
            "kpi_change_summary": "...",
            "final_explanation": "..."
        }
```

---

## 6. ERP 데이터베이스 스키마

### 6.1 Master Data

| Table | Description | Primary Key |
|-------|-------------|-------------|
| MD_PRODUCT | 제품 마스터 | PRODUCT_ID |
| MD_ORG | 조직(법인) 마스터 | ORG_ID |
| MD_CHANNEL | 채널 마스터 | CHANNEL_ID |

**MD_PRODUCT 컬럼**:
- PRODUCT_ID, MODEL_NAME, SERIES, PANEL_TYPE, SCREEN_SIZE
- DISPLAY_TYPE (OLED/LCD), IS_PREMIUM (Y/N), LAUNCH_YEAR

**MD_ORG 컬럼**:
- ORG_ID, ORG_NAME, REGION (Americas/Europe/Asia), CURRENCY

### 6.2 Transaction Data

| Table | Description | Primary Key |
|-------|-------------|-------------|
| TR_SALES | 판매 트랜잭션 | SALES_ID |
| TR_PURCHASE | 구매/원가 트랜잭션 | PURCHASE_ID |
| TR_EXPENSE | 비용 트랜잭션 | EXPENSE_ID |

**TR_SALES 컬럼**:
- SALES_ID, SALES_DATE, ORG_ID, PRODUCT_ID, CHANNEL_ID
- QTY, REVENUE_USD, GROSS_PROFIT_USD, WEBOS_REV_USD

**TR_PURCHASE 컬럼**:
- PURCHASE_ID, PURCHASE_DATE, ORG_ID, PRODUCT_ID
- PANEL_PRICE_USD, TOTAL_COGS_USD, RAW_MATERIAL_INDEX

**TR_EXPENSE 컬럼**:
- EXPENSE_ID, EXPENSE_DATE, ORG_ID, PRODUCT_ID
- LOGISTICS_COST, MARKETING_COST, PROMOTION_COST, LABOR_COST

### 6.3 External Data

| Table | Description | Primary Key |
|-------|-------------|-------------|
| EXT_MACRO | 거시경제 지표 | DATA_DATE |
| EXT_MARKET | 시장 지표 | DATA_DATE |

**EXT_MACRO 컬럼**:
- DATA_DATE, EXCHANGE_RATE_KRW_USD, INFLATION_RATE
- INTEREST_RATE, CSI_INDEX (소비자심리지수)

**EXT_MARKET 컬럼**:
- DATA_DATE, TOTAL_SHIPMENT_10K, LGE_MARKET_SHARE, SCFI_INDEX

---

## 7. 디렉토리 구조

```
BI/
├── agents/                          # Multi-Agent System
│   ├── __init__.py
│   ├── base.py                      # BaseAgent, AgentContext, BaseTool
│   ├── orchestrator.py              # 메인 오케스트레이터
│   ├── search_agent.py              # 검색 에이전트
│   │
│   ├── tools/                       # 공용 도구
│   │   ├── __init__.py
│   │   ├── sql_executor.py          # SQL 실행기
│   │   ├── sql_generator.py         # LLM 기반 SQL 생성기
│   │   ├── graph_executor.py        # Neo4j 실행기
│   │   ├── vector_search.py         # Vector 검색 도구
│   │   └── data_analysis.py         # ★ Shapley 기반 기여도 분석
│   │
│   ├── analysis/                    # 분석 에이전트 그룹
│   │   ├── __init__.py
│   │   ├── analysis_agent.py        # 분석 오케스트레이터
│   │   ├── hypothesis_generator.py  # Graph-Enhanced 가설 생성
│   │   ├── hypothesis_validator.py  # ★ Tier + Shapley 검증
│   │   ├── event_matcher.py         # Hybrid Scoring 이벤트 매칭
│   │   └── evidence_collector.py    # (deprecated)
│   │
│   └── report/                      # 리포트 에이전트
│       └── report_agent.py
│
├── knowledge_graph/                 # Knowledge Graph 구축
│   ├── __init__.py
│   ├── config.py                    # Neo4j 설정
│   │
│   ├── layer1/                      # 3NF RDB Anchoring
│   │   ├── models.py
│   │   ├── dimension_extractor.py
│   │   ├── graph_builder.py
│   │   └── main.py
│   │
│   ├── layer2/                      # Factor-Anchor 관계
│   │   ├── models.py
│   │   ├── factor_extractor.py      # LLM 기반 Factor 추출
│   │   ├── factor_relation_extractor.py
│   │   ├── normalizer.py            # Factor 정규화
│   │   ├── neo4j_loader.py
│   │   └── main.py
│   │
│   ├── layer3/                      # Event Layer
│   │   ├── models.py
│   │   ├── serper_client.py         # 뉴스 검색 API
│   │   ├── search_client.py
│   │   ├── event_extractor.py       # LLM 기반 Event 추출
│   │   ├── event_linker.py          # Event → Factor 연결
│   │   ├── vector_store.py          # Embedding 생성
│   │   ├── neo4j_loader.py
│   │   └── main.py
│   │
│   ├── hybrid_search/               # 하이브리드 검색
│   │   ├── graph_searcher.py
│   │   └── hybrid_engine.py
│   │
│   ├── migrations/                  # 스키마 마이그레이션
│   └── schema/                      # 스키마 정의
│
├── intent_classifier/               # Intent 분류기
│   ├── app.py
│   ├── db_schema.py                 # DB 스키마 정의 (LLM용)
│   └── src/
│       ├── intent_classifier.py
│       └── agent_orchestrator.py
│
├── erp_database/                    # ERP 데이터 생성
│   └── generate_erp_data.py
│
├── app.py                           # ★ Streamlit UI
├── requirements.txt
├── .env                             # 환경변수
└── PROJECT_CONTEXT.md               # 이 파일
```

---

## 8. 환경 설정

### 8.1 필수 의존성

```txt
# requirements.txt

# Core
openai>=1.0.0
neo4j>=5.0.0
streamlit>=1.28.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# ML & Analysis (Shapley 분석용)
scikit-learn>=1.0.0
shap>=0.42.0

# PDF processing
PyMuPDF>=1.23.0

# API clients
requests>=2.31.0

# Environment
python-dotenv>=1.0.0

# YAML processing
pyyaml>=6.0.0

# Visualization
streamlit-echarts>=0.4.0
```

### 8.2 환경변수 (.env)

```bash
OPENAI_API_KEY=sk-xxx
SERPER_API_KEY=xxx
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
DB_PATH=/Users/hyeongrokoh/BI/erp_database/lge_he_erp.db
```

### 8.3 Neo4j Setup (Docker)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest
```

---

## 9. 실행 방법

### 9.1 Streamlit UI 실행

```bash
cd /Users/hyeongrokoh/BI
streamlit run app.py
```

### 9.2 Knowledge Graph 빌드

```bash
# Layer 1: RDB Schema → Graph
python -m knowledge_graph.layer1.main

# Layer 2: Factor 추출 및 로드
python -m knowledge_graph.layer2.main

# Layer 3: Event 추출 및 로드
python -m knowledge_graph.layer3.main
```

### 9.3 ERP 데이터 생성

```bash
python erp_database/generate_erp_data.py
```

---

## 10. Neo4j Graph 통계

### 노드 수
| Label | Count | Description |
|-------|-------|-------------|
| Anchor | 3 | 매출, 원가, 판매수량 |
| Factor/Driver | 464 | 외부 영향 요인 |
| Event | 100+ | 뉴스 기반 이벤트 |
| Dimension | 10+ | Region, ProductCategory, TimePeriod |

### 관계 수
| Type | Count | Description |
|------|-------|-------------|
| AFFECTS | 74+ | Factor/Event → Anchor/Driver |
| INFLUENCES | 451 | Factor → Factor |
| HYPOTHESIZED_TO_AFFECT | 50+ | Driver → KPI |
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

## 11. 주요 Cypher 쿼리

### 가설 생성 시 Factor 조회

```cypher
MATCH (f:Factor)-[r:AFFECTS]->(a:Anchor {id: "revenue"})
RETURN f.name, r.type, r.mention_count
ORDER BY r.mention_count DESC LIMIT 10
```

### Event → Factor → KPI 경로 조회

```cypher
MATCH (e:Event)-[r1:AFFECTS]->(d:Driver {id: $driver_id})
OPTIONAL MATCH (e)-[:TARGETS]->(dim)
WITH e, r1, d, collect(DISTINCT dim.id) as target_dimensions
WHERE size($dimension_filters) = 0
   OR any(df IN $dimension_filters WHERE df IN target_dimensions)
RETURN e.id, e.name, e.severity, r1.polarity, r1.weight
ORDER BY abs(r1.weight) DESC
LIMIT 10
```

### Hybrid Vector + Graph 검색

```cypher
MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor)
WHERE f.name CONTAINS $factor_keyword
WITH e, r, f,
     vector.similarity.cosine(e.embedding, $query_embedding) AS similarity
WHERE similarity > 0.5
RETURN e.name, type(r), r.magnitude, e.severity, similarity
ORDER BY similarity DESC
LIMIT 5
```

---

## 12. 분석 결과 예시

### 입력

```
질문: "2024년 4분기 북미 지역의 매출이 감소한 원인을 분석해줘"
```

### 출력

```
[Step 0] KPI 변동 계산
  매출: 1,200,000 → 960,000 (-20.0%)
  비교 기간: 2024년 Q4 vs 2023년 Q4 (NA)

[Step 1] 가설 생성: 6개 (Graph-Enhanced)

[Step 2] 가설 검증 (Tier + Shapley)
  - [H1] 환율: T1 검증, 기여도 42.3% (#1위) ✅
  - [H2] 할인율: T1 검증, 기여도 28.1% (#2위) ✅
  - [H3] 글로벌수요: T2 검증, 기여도 15.2% (#3위) ✅
  - [H4] 패널원가: T1 검증, 기여도 8.7% (#4위) ❌

[Step 3] 이벤트 매칭: 10개 (Score 0.65-0.82)

[Step 4] 추론 결과
### 분석 결론
2024년 4분기 북미 매출 감소의 주요 원인은 달러 강세(42%), 프로모션 비용 증가(28%),
글로벌 TV 수요 둔화(15%)입니다.

### 상세 분석
1. 환율 상승과 가격 경쟁력 약화
   원/달러 환율이 1,350원을 돌파하면서 미국 시장에서 가격 경쟁력이 약화되었습니다 [1].

2. 프로모션 비용 증가
   블랙프라이데이 시즌 할인 경쟁 심화로 프로모션 비용이 전년 대비 28% 증가했습니다.

### 출처
[1] Reuters: USD/KRW Exchange Rate Hits New High
```

---

## 13. Future Enhancements

1. **Real-time Event Ingestion**: 뉴스 API 연동으로 실시간 Event 추가
2. **Confidence Calibration**: Hybrid Score 가중치 최적화
3. **Multi-hop Reasoning**: Event → Factor → Factor → Anchor 체인 추론
4. **Shapley Waterfall Chart**: 기여도 워터폴 시각화
5. **Feedback Loop**: 사용자 피드백으로 모델 개선
6. **Time-series Forecasting**: 향후 KPI 예측 기능
