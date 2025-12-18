# Knowledge Graph Context Documentation

LG전자 HE(Home Entertainment) 사업부 비즈니스 인텔리전스를 위한 3-Layer Knowledge Graph 시스템

## 프로젝트 개요

### 목적
- TV 사업의 매출/원가/판매수량에 영향을 미치는 요인들의 인과관계 모델링
- 자연어 질문에 대한 인과관계 기반 답변 제공
- 실시간 외부 이벤트가 사업에 미치는 영향 추적

---

## 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
│              "2024년 3분기 매출이 왜 떨어졌어?"                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Intent Classifier                            │
│   intent_classifier/src/intent_classifier.py                     │
│                                                                  │
│   Output:                                                        │
│   - intent: Data QA / Report Generation / Out-of-Scope          │
│   - sub_intent: Internal / External / Hybrid Data               │
│   - analysis_mode: Descriptive / Diagnostic                     │
│   - extracted_entities: {company, period, region, metric...}    │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  SQL Agent      │  │ Knowledge Graph │  │   Hybrid Search     │
│  (Internal)     │  │ (External/Diag) │  │   (Combined)        │
│                 │  │                 │  │                     │
│ lge_he_erp.db   │  │ Neo4j           │  │ SQL + Graph + Vector│
│                 │  │ Layer 1-2-3     │  │                     │
│ - Revenue       │  │ - Events        │  │                     │
│ - Cost          │  │ - Factors       │  │                     │
│ - Sales         │  │ - Causal paths  │  │                     │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Response Generator                          │
│                 (LLM으로 최종 답변 생성)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 데이터 소스 라우팅

| analysis_mode | sub_intent | 데이터 소스 | 예시 질문 |
|---------------|------------|-------------|----------|
| Descriptive | Internal Data | **SQL Agent** (ERP DB) | "2024 Q3 매출 얼마야?" |
| Descriptive | External Data | **Vector Search** (Events) | "홍해 사태 언제 시작됐어?" |
| Diagnostic | Internal Data | **Graph** (Factor→Anchor) | "환율이 원가에 어떤 영향?" |
| Diagnostic | External Data | **Graph** (Event→Factor) | "관세가 왜 올랐어?" |
| Diagnostic | Hybrid Data | **SQL + Graph** | "매출이 왜 떨어졌어?" |

---

## 데이터베이스 구조

### 1. ERP Database (sql/lge_he_erp.db)
**정형 데이터 - SQL Agent가 조회**

```
Tables:
├── TBL_ORG_SUBSIDIARY    # 법인 정보 (LGEUS, LGECA, LGEKR...)
├── TBL_ORG_CUSTOMER      # 고객사 (Best Buy, Amazon...)
├── TBL_MD_PRODUCT        # 제품 마스터 (OLED, QNED...)
├── TBL_TX_SALES_HEADER   # 판매 헤더
├── TBL_TX_SALES_ITEM     # 판매 상세 (수량, 금액)
├── TBL_TX_PRICE_CONDITION # 가격 조건 (할인, MDF, PP)
└── TBL_TX_COST_DETAIL    # 원가 상세 (재료비, 물류비, 관세)
```

**주요 메트릭:**
- Revenue (매출): TBL_TX_SALES_ITEM + TBL_TX_PRICE_CONDITION
- Cost (원가): TBL_TX_COST_DETAIL (MAT, LOG, TAR, OH)
- Sales Quantity (판매수량): TBL_TX_SALES_ITEM

### 2. Knowledge Graph (Neo4j)
**비정형/인과관계 데이터 - Graph + Vector 조회**

```
Nodes:
├── (:Anchor)     # 3개: 매출, 원가, 판매수량
├── (:Factor)     # 464개: 영향 요인
├── (:Event)      # 64개: 외부 이벤트
└── (:Dimension)  # Region, ProductCategory

Relationships:
├── AFFECTS      # Factor → Anchor
├── INFLUENCES   # Factor → Factor
├── INCREASES    # Event → Factor
├── DECREASES    # Event → Factor
└── TARGETS      # Event → Dimension
```

---

## Knowledge Graph 상세
```
┌─────────────────────────────────────────────────────────────────┐
│                    User Query                                    │
│         "환율이 매출에 어떤 영향을 줘?"                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Intent Classifier                              │
│   - Report Generation / Data QA / Out-of-Scope                  │
│   - Descriptive (단순 조회) / Diagnostic (인과 분석)              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────┐         ┌─────────────────────────────────┐
│   Vector Search     │         │      Graph Traversal            │
│   (사실 기반 질의)    │         │      (인과관계 질의)             │
│                     │         │                                 │
│ "홍해 사태 언제?"    │         │ "환율→매출 경로는?"              │
└─────────────────────┘         └─────────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Neo4j Database                              │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Layer 3    │    │  Layer 2    │    │  Layer 1    │         │
│  │   Event     │───▶│   Factor    │───▶│   Anchor    │         │
│  │  (64개)     │    │  (464개)    │    │   (3개)     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                  │                                     │
│        │ TARGETS          │ INFLUENCES                          │
│        ▼                  ▼                                     │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │  Dimension  │    │   Factor    │                            │
│  │ (Region/    │    │  (상호영향)  │                            │
│  │  Product)   │    │             │                            │
│  └─────────────┘    └─────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 구조

### Layer 1: Anchor (핵심 지표)
비즈니스의 최종 목표 지표 3개

| Anchor | 설명 | 영향 Factor 수 |
|--------|------|---------------|
| 매출 (Revenue) | 총 매출액 | 30+ |
| 원가 (Cost) | 제조/물류 원가 | 25+ |
| 판매수량 (Sales Volume) | 총 판매 대수 | 20+ |

### Layer 2: Factor (영향 요인)
Anchor에 영향을 주는 464개의 요인들

**카테고리별 분포:**
- Cost Factors: 물류비, 관세, 패널가격, 환율 등
- Revenue Factors: 가격, 프리미엄 믹스, 채널 등
- Volume Factors: 수요, 시장점유율, 경쟁 등
- External Factors: 거시경제, 규제, 기술 등

**관계 유형:**
- `AFFECTS`: Factor → Anchor (직접 영향)
  - 영향 방향: PROPORTIONAL / INVERSE
  - 예: 환율 ─PROPORTIONAL→ 원가
- `INFLUENCES`: Factor → Factor (간접 영향)
  - 예: 유가 → 물류비 → 원가

### Layer 3: Event (외부 이벤트)
실시간 뉴스에서 추출한 64개의 이벤트

**카테고리별 분포:**
| 카테고리 | 수량 | 예시 |
|---------|------|------|
| company | 25 | LG 마이크로RGB 출시, BOE OLED 확장 |
| policy | 13 | 트럼프 관세, 텍사스 개인정보 소송 |
| market | 12 | LCD 패널 가격 최저, 블랙프라이데이 |
| technology | 7 | 마이크로RGB, AI TV |
| macro_economy | 5 | 원/달러 1480원 돌파 |
| geopolitical | 2 | 가자 휴전, 한미 관세협상 |

**관계 유형:**
- `INCREASES`: Event → Factor (증가 영향)
- `DECREASES`: Event → Factor (감소 영향)
- `TARGETS`: Event → Dimension (영향 범위)

---

## 데이터 흐름

### 1. 뉴스 수집 (Brave Search API)
```
60개 쿼리 × 20개 결과 = 1,200개 뉴스
    ↓ 중복 제거
906개 고유 뉴스
```

### 2. 이벤트 추출 (GPT-4o)
```
906개 뉴스 → LLM 추출 → 69개 Event
    ↓ 정규화/중복 제거
64개 고유 Event
```

### 3. 관계 생성
```
Event → Factor: 115개 (INCREASES: 77, DECREASES: 38)
Event → Dimension: 98개 (TARGETS)
```

### 4. Vector Embedding (OpenAI)
```
64개 Event → 123개 청크 → 임베딩 저장
```

---

## Neo4j 스키마

### 노드 레이블
```cypher
(:Anchor)     -- 3개: 매출, 원가, 판매수량
(:Factor)     -- 464개: 영향 요인
(:Event)      -- 64개: 외부 이벤트
(:Dimension)  -- Region, ProductCategory
```

### 관계 유형
```cypher
// Layer 2 → Layer 1
(:Factor)-[:AFFECTS {direction, weight}]->(:Anchor)

// Layer 2 내부
(:Factor)-[:INFLUENCES {direction, weight}]->(:Factor)

// Layer 3 → Layer 2
(:Event)-[:INCREASES {magnitude, confidence}]->(:Factor)
(:Event)-[:DECREASES {magnitude, confidence}]->(:Factor)

// Layer 3 → Dimension
(:Event)-[:TARGETS {specificity}]->(:Dimension)
```

### 유용한 쿼리
```cypher
// 1. 특정 Anchor에 영향을 주는 모든 Factor
MATCH (f:Factor)-[:AFFECTS]->(a:Anchor {name: '매출'})
RETURN f.name, f.category

// 2. Event → Factor → Anchor 전체 경로
MATCH path = (e:Event)-[:INCREASES|DECREASES]->(f:Factor)-[:AFFECTS]->(a:Anchor)
RETURN e.name, f.name, a.name

// 3. 특정 지역에 영향 주는 이벤트
MATCH (e:Event)-[:TARGETS]->(d:Dimension {name: 'NA'})
RETURN e.name, e.category, e.severity

// 4. Factor 간 영향 체인
MATCH path = (f1:Factor)-[:INFLUENCES*1..3]->(f2:Factor)-[:AFFECTS]->(a:Anchor)
WHERE f1.name = '유가'
RETURN path

// 5. Vector 유사도 검색 (Neo4j 5.11+)
CALL db.index.vector.queryNodes('event_embedding', 5, $queryVector)
YIELD node, score
RETURN node.name, score
```

---

## 파일 구조

```
knowledge_graph/
├── context/                    # 문서화
│   └── README.md              # 이 파일
│
├── layer1/                    # Anchor Layer
│   ├── models.py             # Anchor 모델 정의
│   └── neo4j_loader.py       # Neo4j 적재
│
├── layer2/                    # Factor Layer
│   ├── models.py             # Factor 모델
│   ├── factor_taxonomy.yaml  # Factor 분류 체계
│   ├── factor_extractor.py   # 애널리스트 리포트에서 추출
│   ├── normalizer.py         # 정규화/중복 제거
│   └── neo4j_loader.py       # Neo4j 적재
│
├── layer3/                    # Event Layer
│   ├── models.py             # Event, EventFactorRelation 등
│   ├── config.py             # API 키, CORE_FACTORS
│   ├── search_queries.yaml   # 60개 검색 쿼리
│   ├── event_normalization.yaml  # 이벤트 정규화 사전
│   ├── search_client.py      # Brave Search API
│   ├── event_extractor.py    # LLM 기반 이벤트 추출
│   ├── normalizer.py         # 중복 제거
│   ├── vector_store.py       # Vector Embedding
│   ├── neo4j_loader.py       # Neo4j 적재
│   ├── main.py               # CLI (build, load, full)
│   └── layer3_result.json    # 최종 결과
│
└── (향후 추가 예정)
    ├── hybrid_search.py      # Graph + Vector 하이브리드 검색
    └── query_engine.py       # 자연어 질의 처리
```

---

## 사용법

### Layer 3 빌드
```bash
# 전체 파이프라인 (뉴스 수집 → 이벤트 추출 → 정규화 → 벡터)
python -m knowledge_graph.layer3.main build --max-queries 60 --results-per-query 20

# Neo4j 적재
python -m knowledge_graph.layer3.main load

# 전체 (build + load)
python -m knowledge_graph.layer3.main full
```

### Python API
```python
from knowledge_graph.layer3 import build_layer3, load_layer3_to_neo4j

# 빌드
graph = build_layer3(max_queries=60, results_per_query=20)

# 적재
load_layer3_to_neo4j(graph, clear_existing=True)
```

---

## 향후 계획

### Phase 1: Hybrid Search (진행 예정)
- Graph 쿼리 + Vector 검색 결합
- Intent Classifier 연동
- Descriptive → Vector Search
- Diagnostic → Graph Traversal

### Phase 2: Simulation Engine
- What-if 분석 (이벤트 발생 시 영향 시뮬레이션)
- Factor 전파 경로 계산

### Phase 3: Auto-Update Pipeline
- 주기적 뉴스 수집 (daily/weekly)
- 새 Event 자동 추출 및 적재
- 고위험 이벤트 알림

---

## 의존성

```
neo4j>=5.0
openai>=1.0
pyyaml
requests
```

## 환경 변수
```
OPENAI_API_KEY=...
BRAVE_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```
