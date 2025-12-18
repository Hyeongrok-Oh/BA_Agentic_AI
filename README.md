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

## Project Structure

```
BI/
├── app.py                      # Streamlit 메인 애플리케이션
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
├── knowledge_graph/
│   ├── layer1/                 # Anchor & Dimension
│   ├── layer2/                 # Factor
│   ├── layer3/                 # Event
│   └── hybrid_search/          # Graph + Vector 검색
├── sql/                        # SQL Agent & Mock 데이터
├── intent_classifier/          # Intent Classification
└── docs/                       # 기술 문서
```

## Installation

### Option 1: Docker (Recommended)

Docker를 사용하면 Neo4j와 애플리케이션을 한 번에 실행할 수 있습니다.

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/multi-agent-bi.git
cd multi-agent-bi

# 2. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY를 입력하세요

# 3. Docker Compose로 실행
docker-compose up -d

# 4. Neo4j 초기 데이터 로드 (최초 1회)
docker exec -it bi-app python scripts/init_neo4j.py

# 5. 브라우저에서 접속
# App: http://localhost:8501
# Neo4j Browser: http://localhost:7474 (neo4j/password123)
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/multi-agent-bi.git
cd multi-agent-bi

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Neo4j 별도 설치 필요 (https://neo4j.com/download/)
```

## Environment Variables

```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
BRAVE_API_KEY=your_brave_api_key  # Optional
```

## Usage

### Docker
```bash
# 시작
docker-compose up -d

# 종료
docker-compose down

# 로그 확인
docker-compose logs -f app
```

### Local
```bash
# Run Streamlit app
streamlit run app.py
```

## Example Query

```
"2025년 Q4 북미 매출 감소 원인은?"
```

시스템이 자동으로:
1. KPI 변동 계산 (-24.0%)
2. Knowledge Graph에서 가설 생성 (74개)
3. ERP 데이터로 가설 검증 (12개 유효)
4. 관련 이벤트 매칭 (뉴스 기반)
5. Top 3 원인 분석 및 보고서 생성

## Documentation

자세한 기술 문서는 [docs/report_section_3_4.md](docs/report_section_3_4.md)를 참조하세요.

## License

MIT License
