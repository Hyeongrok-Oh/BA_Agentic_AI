# Multi-Agent BI System

LG전자 HE(Home Entertainment) 사업부의 KPI 변동 원인을 자동으로 분석하는 Multi-Agent BI 시스템입니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Intent Classifier Agent                        │
│            (사용자 질문 분석 → JSON 형태로 의도 분류)               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Orchestrator                              │
├─────────────────────────────────────────────────────────────────┤
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│    │Data Extraction│    │   분석 Agent  │    │보고서 생성 Agent│   │
│    │    Agent      │◄───│              │    │   (구현중)     │    │
│    └──────┬───────┘    └──────────────┘    └──────────────┘     │
│           │                                                      │
│    ┌──────┴──────────────────────┐                              │
│    │         Tools (공유)          │                              │
│    ├─────────┬─────────┬─────────┤                              │
│    │SQL Agent│GraphRAG │VectorRAG│                              │
│    │         │  Agent  │  Agent  │                              │
│    └─────────┴─────────┴─────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## Knowledge Graph Structure

3계층 Knowledge Graph를 사용하여 KPI 변동 원인을 분석합니다:

- **Layer 1 (Anchor & Dimension)**: 매출, 원가, 판매수량 + 지역/제품/채널
- **Layer 2 (Factor)**: 464개 Factor, 74개 AFFECTS, 451개 INFLUENCES 관계
- **Layer 3 (Event)**: 69개 Event, 86개 INCREASES, 32개 DECREASES 관계

## Features

- 자연어 질문으로 KPI 변동 원인 분석
- Knowledge Graph 기반 가설 생성
- SQL 기반 가설 검증
- 뉴스/이벤트 기반 근거 강화
- 경영진 보고서 형태 결과 출력

## Tech Stack

- **LLM**: OpenAI GPT-4o, o1
- **Knowledge Graph**: Neo4j
- **Vector Search**: Neo4j Vector Index + OpenAI Embeddings
- **ERP Database**: SQLite
- **Frontend**: Streamlit
- **News Search**: Brave Search API

---

## Quick Start (Docker)

### 사전 요구사항

1. **Docker Desktop** 설치
   - Mac: https://docs.docker.com/desktop/install/mac-install/
   - Windows: https://docs.docker.com/desktop/install/windows-install/

2. **OpenAI API Key** 준비
   - https://platform.openai.com/api-keys 에서 발급

### Step 1: 저장소 클론

```bash
git clone https://github.com/Hyeongrok-Oh/BA_Agentic_AI.git
cd BA_Agentic_AI
```

### Step 2: 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 파일을 열어 OpenAI API Key를 입력합니다:

```bash
# Mac/Linux
nano .env

# Windows
notepad .env
```

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx  # 실제 API 키 입력
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

### Step 3: Docker 실행

```bash
# Docker 컨테이너 시작 (백그라운드)
docker-compose up -d
```

실행되는 컨테이너:
- `bi-neo4j`: Neo4j 데이터베이스 (포트 7474, 7687)
- `bi-app`: Streamlit 애플리케이션 (포트 8501)

### Step 4: Neo4j 초기화 (최초 1회만)

Neo4j가 완전히 시작될 때까지 약 30초 대기 후:

```bash
# Knowledge Graph 데이터 로드
docker exec -it bi-app python scripts/init_neo4j.py
```

성공 시 출력:
```
=== Summary ===
  Factor: 16 nodes
  Anchor: 3 nodes
  Region: 6 nodes
  ...
✅ Neo4j initialization complete!
```

### Step 5: 애플리케이션 접속

브라우저에서 접속:
- **앱**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474

### 종료 및 재시작

```bash
# 종료
docker-compose down

# 재시작 (데이터 유지됨)
docker-compose up -d

# 로그 확인
docker-compose logs -f app

# 컨테이너 상태 확인
docker-compose ps
```

---

## Neo4j Knowledge Graph 조회

### Neo4j Browser 접속

1. http://localhost:7474 접속
2. 로그인 정보 입력:
   - **Username**: `neo4j`
   - **Password**: `password123`

### 기본 Cypher 쿼리

#### 전체 노드/관계 통계
```cypher
// 노드 유형별 개수
MATCH (n)
RETURN labels(n)[0] as NodeType, count(*) as Count
ORDER BY Count DESC

// 관계 유형별 개수
MATCH ()-[r]->()
RETURN type(r) as RelationType, count(*) as Count
ORDER BY Count DESC
```

#### Layer 1: Anchor & Dimension 조회
```cypher
// 모든 Anchor (KPI) 조회
MATCH (a:Anchor)
RETURN a.id, a.name, a.name_en, a.unit

// 모든 Region 조회
MATCH (r:Region)
RETURN r.id, r.name, r.name_en

// 모든 ProductCategory 조회
MATCH (p:ProductCategory)
RETURN p.id, p.name, p.is_premium
```

#### Layer 2: Factor 조회
```cypher
// 모든 Factor 조회
MATCH (f:Factor)
RETURN f.id, f.name, f.category, f.direction
ORDER BY f.category

// 특정 카테고리 Factor 조회 (예: logistics)
MATCH (f:Factor {category: 'logistics'})
RETURN f.name, f.direction

// Factor → Anchor 관계 (AFFECTS)
MATCH (f:Factor)-[r:AFFECTS]->(a:Anchor)
RETURN f.name as Factor, r.direction as Direction, a.name as Anchor

// Factor → Factor 관계 (INFLUENCES)
MATCH (f1:Factor)-[r:INFLUENCES]->(f2:Factor)
RETURN f1.name as From, r.direction as Direction, f2.name as To
```

#### Layer 3: Event 조회
```cypher
// 모든 Event 조회
MATCH (e:Event)
RETURN e.id, e.name, e.category, e.severity, e.is_ongoing
ORDER BY e.severity DESC

// 특정 카테고리 Event (예: geopolitical)
MATCH (e:Event {category: 'geopolitical'})
RETURN e.name, e.evidence

// Event → Factor 관계 (INCREASES/DECREASES)
MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor)
RETURN e.name as Event, type(r) as Impact, f.name as Factor, r.magnitude

// Event → Dimension 관계 (TARGETS)
MATCH (e:Event)-[:TARGETS]->(d)
RETURN e.name as Event, labels(d)[0] as DimensionType, d.name as Dimension
```

#### 인과관계 경로 탐색
```cypher
// Event → Factor → Anchor 전체 경로
MATCH path = (e:Event)-[:INCREASES|DECREASES]->(f:Factor)-[:AFFECTS]->(a:Anchor)
RETURN e.name as Event, f.name as Factor, a.name as KPI
LIMIT 20

// 특정 KPI(매출)에 영향을 주는 모든 경로
MATCH path = (e:Event)-[:INCREASES|DECREASES]->(f:Factor)-[:AFFECTS]->(a:Anchor {name: '매출'})
RETURN e.name as Event,
       CASE WHEN type(relationships(path)[0]) = 'INCREASES' THEN '증가' ELSE '감소' END as EventImpact,
       f.name as Factor,
       a.name as KPI

// 북미(NA) 지역에 영향을 주는 Event
MATCH (e:Event)-[:TARGETS]->(r:Region {id: 'NA'})
MATCH (e)-[rel:INCREASES|DECREASES]->(f:Factor)
RETURN e.name as Event, type(rel) as Impact, f.name as Factor
```

#### 그래프 시각화
```cypher
// 전체 Knowledge Graph 시각화 (노드 50개 제한)
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 50

// Layer 2-3 관계만 시각화
MATCH (e:Event)-[r1]->(f:Factor)-[r2]->(a:Anchor)
RETURN e, r1, f, r2, a
LIMIT 30
```

---

## Project Structure

```
BA_Agentic_AI/
├── app.py                      # Streamlit 메인 애플리케이션
├── Dockerfile                  # Docker 이미지 빌드
├── docker-compose.yml          # Docker Compose 설정
├── requirements.txt            # Python 패키지
├── .env.example                # 환경변수 템플릿
├── QUICKSTART.md               # 빠른 시작 가이드
│
├── scripts/
│   └── init_neo4j.py           # Neo4j 초기화 스크립트
│
├── agents/
│   ├── base.py                 # BaseAgent, AgentContext
│   ├── orchestrator.py         # Orchestrator
│   ├── analysis/               # 분석 Agent
│   │   ├── analysis_agent.py
│   │   ├── hypothesis_generator.py
│   │   ├── hypothesis_validator.py
│   │   └── event_matcher.py
│   └── tools/                  # Tool Agents
│       ├── sql_generator.py
│       ├── sql_executor.py
│       ├── graph_executor.py
│       └── vector_search.py
│
├── knowledge_graph/
│   ├── layer1/                 # Anchor & Dimension
│   ├── layer2/                 # Factor (추출된 JSON 포함)
│   ├── layer3/                 # Event
│   └── hybrid_search/          # Graph + Vector 검색
│
├── sql/                        # SQL Agent & Mock ERP 데이터
├── intent_classifier/          # Intent Classification
└── docs/                       # 기술 문서
```

---

## Example Query

```
"2025년 4분기 북미 매출 감소 원인은?"
```

시스템이 자동으로:
1. KPI 변동 계산 (-24.0%)
2. Knowledge Graph에서 가설 생성 (74개)
3. ERP 데이터로 가설 검증 (12개 유효)
4. 관련 이벤트 매칭 (뉴스 기반)
5. Top 3 원인 분석 및 보고서 생성

---

## Troubleshooting

### Docker가 실행되지 않는 경우
```bash
# Docker Desktop이 실행 중인지 확인
docker ps

# Docker Desktop 재시작 후 다시 시도
docker-compose up -d
```

### Neo4j 연결 실패
```bash
# Neo4j 로그 확인
docker-compose logs neo4j

# Neo4j가 완전히 시작될 때까지 대기 (약 30초)
# 이후 init_neo4j.py 재실행
docker exec -it bi-app python scripts/init_neo4j.py
```

### 앱이 느린 경우
- Docker Desktop → Settings → Resources
- Memory를 4GB 이상으로 설정

### 포트 충돌
```bash
# 사용 중인 포트 확인
lsof -i :8501
lsof -i :7474

# 기존 컨테이너 정리
docker-compose down
docker system prune -f
```

---

## Documentation

자세한 기술 문서는 [docs/report_section_3_4.md](docs/report_section_3_4.md)를 참조하세요.

## License

MIT License
