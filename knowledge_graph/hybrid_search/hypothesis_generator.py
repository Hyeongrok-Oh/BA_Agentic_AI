"""
가설 생성기 - KPI 변동 원인에 대한 가설 생성
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class Hypothesis:
    """가설 데이터 클래스"""
    id: str
    category: str  # revenue, cost, volume, external
    factor: str  # 관련 Factor 이름
    direction: str  # increase, decrease
    description: str  # 가설 설명
    sql_check: str  # 검증용 SQL 힌트
    validated: Optional[bool] = None
    validation_data: Optional[Dict] = None


HYPOTHESIS_PROMPT = """당신은 LG전자 HE(Home Entertainment) 사업부의 재무 분석 전문가입니다.

## 사용자 질문
{question}

## 추출된 정보
- 회사: {company}
- 기간: {period}
- 지역: {region}
- KPI: {kpi}

## 데이터베이스 구조
**ERP 테이블:**
- TBL_TX_SALES_HEADER: 판매 헤더 (DOC_DATE, SUBSIDIARY_ID, TOTAL_NET_VALUE)
- TBL_TX_SALES_ITEM: 판매 상세 (ORDER_QTY, NET_VALUE)
- TBL_TX_COST_DETAIL: 원가 상세
  - COST_TYPE: MAT(재료비), LOG(물류비), TAR(관세), OH(오버헤드)
- TBL_TX_PRICE_CONDITION: 가격 조건
  - COND_TYPE: ZPR0(매출), ZPRO(Price Protection), K007(할인), ZMDF(MDF)

**지역 매핑:**
- 북미: SUBSIDIARY_ID IN ('LGEUS', 'LGECA')
- 유럽: SUBSIDIARY_ID IN ('LGEUK', 'LGEDE', 'LGEFR')
- 한국: SUBSIDIARY_ID = 'LGEKR'

## 태스크
KPI 변동의 가능한 원인에 대해 5-7개의 가설을 생성하세요.
각 가설은 SQL로 검증 가능해야 합니다.

## 가설 카테고리
1. **revenue**: 매출 관련 (판매량, 가격, 믹스 등)
2. **cost**: 원가 관련 (재료비, 물류비, 관세, 오버헤드)
3. **pricing**: 가격 조건 관련 (할인, PP, MDF)
4. **external**: 외부 요인 (환율, 경쟁, 시장)

## 응답 형식 (JSON)
```json
{{
  "hypotheses": [
    {{
      "id": "H1",
      "category": "cost",
      "factor": "물류비",
      "direction": "increase",
      "description": "물류비 증가로 인한 영업이익 감소",
      "sql_check": "COST_TYPE = 'LOG' 비교"
    }},
    {{
      "id": "H2",
      "category": "pricing",
      "factor": "Price Protection",
      "direction": "increase",
      "description": "프로모션 비용(PP) 증가로 인한 수익성 악화",
      "sql_check": "COND_TYPE = 'ZPRO' 비교"
    }}
  ]
}}
```
"""


class HypothesisGenerator:
    """KPI 변동 원인 가설 생성기"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 필요")
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        question: str,
        company: str = "LGE",
        period: str = None,
        region: str = None,
        kpi: str = "영업이익"
    ) -> List[Hypothesis]:
        """가설 생성"""

        prompt = HYPOTHESIS_PROMPT.format(
            question=question,
            company=company,
            period=period or "미지정",
            region=region or "전체",
            kpi=kpi
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 재무 분석 전문가입니다. JSON 형식으로만 응답하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, response: str) -> List[Hypothesis]:
        """LLM 응답 파싱"""
        import re

        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match2 = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match2:
                json_str = json_match2.group(0)
            else:
                json_str = response

        try:
            data = json.loads(json_str)
            hypotheses = []

            for h in data.get("hypotheses", []):
                hypotheses.append(Hypothesis(
                    id=h.get("id", ""),
                    category=h.get("category", ""),
                    factor=h.get("factor", ""),
                    direction=h.get("direction", ""),
                    description=h.get("description", ""),
                    sql_check=h.get("sql_check", "")
                ))

            return hypotheses

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return []

    def generate_from_intent(self, intent_result: Dict) -> List[Hypothesis]:
        """Intent Classifier 결과에서 가설 생성"""
        entities = intent_result.get("extracted_entities", {}) or {}

        # Period 파싱
        period = entities.get("period")
        period_str = None
        if period:
            year = period.get("year")
            quarter = period.get("quarter")
            if year and quarter:
                period_str = f"{year}년 Q{quarter}"
            elif year:
                period_str = f"{year}년"

        # Region 파싱
        region = entities.get("region")
        if isinstance(region, list):
            region = ", ".join(region)

        return self.generate(
            question=intent_result.get("thinking", ""),
            company=entities.get("company", "LGE"),
            period=period_str,
            region=region,
            kpi="영업이익"
        )
