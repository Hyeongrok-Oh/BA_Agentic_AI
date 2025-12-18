# 의도 분류(Intent Classification) Agent - Capston

> **LLM 기반 의도 분류 및 엔티티 추출 시스템**  
> 사용자의 자연어 질문을 계층적으로 분류하고 핵심 정보를 추출합니다.

---

## 📌 프로젝트 개요

이 프로젝트는 **GPT-4o-mini**를 활용하여 사용자의 자연어 질문을 분석하고:
1. **의도(Intent)를 계층적으로 분류** (3-Layer Classification)
2. **핵심 엔티티(회사, 기간, 메트릭 등)를 추출**
3. **부족한 정보에 대해 Clarifying Question 생성**

### 핵심 기능

| 기능 | 설명 |
|------|------|
| **계층적 의도 분류** | Layer 1 → Layer 2 → Layer 3 단계별 분류 |
| **엔티티 추출** | 회사명, 기간, 지역, 메트릭 등 JSON 추출 |
| **Dynamic Few-Shot** | 유사 질문 검색하여 프롬프트에 동적 반영 |
| **Guardrail** | 비즈니스 외 질문 사전 필터링 |
| **Multi-turn 대화** | 이전 대화 맥락 유지 |

---

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# Python 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. OpenAI API Key 설정

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# macOS/Linux
export OPENAI_API_KEY=your-api-key-here
```

### 3. 애플리케이션 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 📁 프로젝트 구조

```
Capston/
├── app.py                    # 🎯 메인 Streamlit 애플리케이션
├── db_schema.py              # 📊 데이터베이스 스키마 (LLM 프롬프트용)
├── requirements.txt          # 📦 Python 의존성 목록
│
├── src/                      # 🔧 핵심 소스 코드
│   ├── intent_classifier.py  # ⭐ 의도 분류 엔진 (핵심)
│   ├── schemas.py            # Pydantic 스키마 정의
│   ├── guardrail.py          # 비즈니스 도메인 필터
│   ├── agent_orchestrator.py # 에이전트 오케스트레이터
│   ├── data/                 # Few-shot 예제 데이터
│   │   └── few_shot_examples.json
│   ├── services/             # 서비스 레이어
│   ├── ui/                   # UI 컴포넌트
│   └── utils/                # 유틸리티 (Embedding 검색 등)
│
└── src/data/                 # Few-shot 예제 데이터
    └── few_shot_examples.json
```

---

## 🏗️ 시스템 아키텍처

### 처리 플로우

```
사용자 입력
    ↓
┌─────────────────┐
│   Guardrail    │ → 비즈니스 외 질문 필터링
└────────┬────────┘
         ↓
┌─────────────────┐
│ Intent Classifier │ → GPT-4o-mini + Dynamic Few-Shot
└────────┬────────┘
         ↓
┌─────────────────┐
│  JSON 출력      │ → Intent + Entities + Clarifying Q
└─────────────────┘
```

### 계층적 의도 분류 (3-Layer)

| Layer | 분류 항목 | 설명 |
|-------|----------|------|
| **Layer 1** | Report Generation, Data QA, Ambiguous | 최상위 의도 |
| **Layer 2** | Defined Report, New Report, Internal/External/Hybrid Data, Data Unavailable, Ambiguous Clarification | 세부 의도 |
| **Layer 3** | Pre-closing, Post-closing, External Event, Required Slot Missing, Metric Unavailable, Date Out of Range | 상세 유형 |

---

## 🔌 다른 프로젝트와 연동 방법

### 방법 1: 모듈로 직접 Import

```python
import sys
sys.path.append('path/to/Capston')

from src.intent_classifier import IntentClassifier

# 초기화
classifier = IntentClassifier(api_key="your-openai-api-key")

# 의도 분류
messages = [
    {"role": "user", "content": "2024년 3분기 북미 매출액 알려줘"}
]
result = classifier.classify(messages)

# 결과 확인
print(result.intent)           # "Data QA"
print(result.sub_intent)       # "Internal Data"
print(result.detail_type)      # "Post-closing"
print(result.extracted_entities)  # ExtractedEntities 객체
```

### 방법 2: 결과 JSON 형식

```json
{
  "thinking": "사용자가 2024년 3분기 북미 매출액을 요청...",
  "intent": "Data QA",
  "sub_intent": "Internal Data",
  "detail_type": "Post-closing",
  "analysis_mode": "Descriptive",
  "extracted_entities": {
    "company": "LG전자",
    "period": {"year": 2024, "quarter": 3},
    "region": "북미",
    "requested_metrics": ["Revenue"]
  },
  "response_message": "2024년 3분기 북미 매출액 조회 중입니다.",
  "recommended_questions": [
    "영업이익도 함께 확인하시겠습니까?",
    "작년 동기 대비 비교도 필요하신가요?"
  ]
}
```

### 방법 3: REST API 서버로 확장 (예시)

```python
from flask import Flask, request, jsonify
from src.intent_classifier import IntentClassifier

app = Flask(__name__)
classifier = IntentClassifier()

@app.route('/classify', methods=['POST'])
def classify_intent():
    data = request.json
    messages = data.get('messages', [])
    result = classifier.classify(messages)
    return jsonify({
        "intent": result.intent,
        "sub_intent": result.sub_intent,
        "detail_type": result.detail_type,
        "entities": result.extracted_entities.dict() if result.extracted_entities else None
    })

if __name__ == '__main__':
    app.run(port=5000)
```

---

## 📊 주요 스키마 (`src/schemas.py`)

### IntentResult
```python
class IntentResult(BaseModel):
    thinking: str                    # 추론 과정
    intent: str                      # Layer 1 (Report Generation/Data QA/Ambiguous)
    sub_intent: SubIntentEnum        # Layer 2
    detail_type: DetailTypeEnum      # Layer 3
    analysis_mode: AnalysisMode      # Descriptive/Diagnostic
    extracted_entities: ExtractedEntities  # 추출된 엔티티
    response_message: str            # 사용자 응답 메시지
    recommended_questions: List[str] # 추천 후속 질문
```

### ExtractedEntities
```python
class ExtractedEntities(BaseModel):
    company: str                    # 회사명
    period: Period                  # 기간 (year, quarter, month)
    region: str                     # 지역
    customer: str                   # 고객사
    product: str                    # 제품
    requested_metrics: List[MetricEnum]  # 요청 메트릭
```

---

## 🧪 테스트 실행

```bash
# 전체 테스트
pytest tests/

# 의도 분류 테스트
python test_layer123_json.py
```

---

## 📈 성능 평가

```bash
# 의도 분류 평가
python evaluation/scripts/evaluate_comprehensive_intent.py
```

---

## 🔑 환경 변수

| 변수명 | 필수 | 설명 |
|--------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI API 키 |

---

## 🛠️ 기술 스택

- **Python** 3.10+
- **LLM**: OpenAI GPT-4o-mini
- **웹**: Streamlit
- **스키마**: Pydantic
- **Embedding**: OpenAI text-embedding-3-small
- **테스트**: pytest

---

## 📞 문의

프로젝트 관련 문의사항이 있으시면 담당자에게 연락해 주세요.
