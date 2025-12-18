# Quick Start Guide

## 사전 요구사항

1. **Docker Desktop** 설치
   - Mac: https://docs.docker.com/desktop/install/mac-install/
   - Windows: https://docs.docker.com/desktop/install/windows-install/

2. **OpenAI API Key** 준비
   - https://platform.openai.com/api-keys 에서 발급

## 실행 방법

### Mac/Linux

```bash
# 1. 프로젝트 폴더로 이동
cd multi-agent-bi

# 2. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 입력

# 3. 실행 (자동)
./start.sh
```

### Windows

```batch
# 1. 프로젝트 폴더로 이동
cd multi-agent-bi

# 2. 실행
start.bat
# 메모장이 열리면 OPENAI_API_KEY 입력 후 저장
```

### 수동 실행

```bash
# Docker 시작
docker-compose up -d

# Neo4j 초기화 (최초 1회)
docker exec bi-app python scripts/init_neo4j.py
```

## 접속

- **애플리케이션**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474 (ID: neo4j, PW: password123)

## 종료

```bash
docker-compose down
```

## 예시 질문

- "2024년 4분기 북미 영업이익이 왜 감소했어?"
- "2025년 Q3 매출 변동 원인 분석해줘"
- "최근 물류 관련 이벤트 알려줘"

## 문제 해결

### Docker가 실행되지 않는 경우
```bash
# Docker Desktop이 실행 중인지 확인
docker ps
```

### Neo4j 연결 실패
```bash
# 로그 확인
docker-compose logs neo4j
```

### 앱이 느린 경우
- Docker Desktop에서 메모리를 4GB 이상으로 설정
