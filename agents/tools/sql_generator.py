"""
SQL Generator - LLM 기반 SQL 쿼리 생성기

역할:
- 자연어 질문을 분석하여 SQL 쿼리 생성
- Chain-of-Thought 추론으로 정확한 쿼리 보장
- SQLExecutor와 분리되어 생성만 담당
"""

import os
import sqlite3
from typing import Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

from ..base import BaseTool, ToolResult


@dataclass
class SQLGenerationResult:
    """SQL 생성 결과"""
    query: str
    reasoning: str
    success: bool
    error: Optional[str] = None


class SQLGenerator(BaseTool):
    """LLM 기반 SQL 쿼리 생성기"""

    name = "sql_generator"
    description = "자연어 질문을 분석하여 SQLite 쿼리를 생성합니다."

    DEFAULT_DB_PATH = "/Users/hyeongrokoh/BI/sql/lge_he_erp.db"

    def __init__(self, db_path: str = None, api_key: str = None):
        self.db_path = db_path or os.getenv("ERP_DB_PATH", self.DEFAULT_DB_PATH)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._schema_cache = None

    @property
    def client(self) -> OpenAI:
        """Lazy OpenAI client initialization"""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def schema_info(self) -> str:
        """데이터베이스 스키마 정보 (캐시)"""
        if self._schema_cache is None:
            self._schema_cache = self._get_schema_info()
        return self._schema_cache

    def _get_schema_info(self) -> str:
        """데이터베이스 스키마 정보 추출"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            schema_lines = ["=== DATABASE SCHEMA ===\n"]

            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                schema_lines.append(f"\nTable: {table_name}")
                schema_lines.append("Columns:")
                for col in columns:
                    col_id, name, col_type, not_null, default_val, pk = col
                    pk_marker = " (PRIMARY KEY)" if pk else ""
                    schema_lines.append(f"  - {name}: {col_type}{pk_marker}")

            conn.close()
            return "\n".join(schema_lines)
        except Exception as e:
            return f"Schema extraction error: {e}"

    def _create_sql_prompt(self, question: str, context: Dict = None) -> str:
        """SQL 생성 프롬프트"""
        context = context or {}
        period = context.get("period", {})
        region = context.get("region")

        period_hint = ""
        if period:
            year = period.get("year", 2024)
            quarter = period.get("quarter", 4)
            period_hint = f"\n현재 분석 기간: {year}년 Q{quarter}"

        region_hint = ""
        if region:
            region_hint = f"\n현재 분석 지역: {region}"

        return f"""당신은 LG전자 HE사업부 ERP 데이터베이스 전문가입니다.

{self.schema_info}

=== 비즈니스 컨텍스트 ===

**지역별 SUBSIDIARY_ID 매핑 (필수)**:
- 북미(NA, North America): SUBSIDIARY_ID IN ('LGEUS', 'LGECA')
- 유럽(EU, Europe): SUBSIDIARY_ID IN ('LGEUK', 'LGEDE', 'LGEFR')
- 한국(KR, Korea): SUBSIDIARY_ID = 'LGEKR'

**원가 유형 (TBL_TX_COST_DETAIL 테이블)**:
- COST_TYPE: 원가 유형
  - MAT: 재료비 (Material Cost)
  - LOG: 물류비 (Logistics Cost)
  - TAR: 관세 (Tariff)
  - OH: 오버헤드 (Overhead)
- COST_AMOUNT: 단위 원가 금액 (이 컬럼 사용!)
- ORDER_NO, ITEM_NO: JOIN 키

**매출 헤더 (TBL_TX_SALES_HEADER 테이블)**:
- ORDER_NO: 주문번호 (PK)
- DOC_DATE: 문서 날짜 (기간 필터용)
- SUBSIDIARY_ID: 법인 코드 (지역 필터용)
- TOTAL_NET_VALUE: 순매출액

**매출 아이템 (TBL_TX_SALES_ITEM 테이블)**:
- ORDER_NO, ITEM_NO: 키
- ORDER_QTY: 주문 수량

**가격 조건 (TBL_TX_PRICE_CONDITION 테이블)**:
- COND_TYPE: 조건 유형
  - ZPR0: 매출 (Gross Price) - 양수
  - ZPRO: Price Protection (비용) - 음수
  - K007: 할인 (Discount) - 음수
  - ZMDF: MDF (마케팅비용) - 음수
- COND_VALUE: 금액 (이 컬럼 사용!)
- ORDER_NO: 주문번호 (JOIN 키)

**기간 필터**:
- Q1: 01-03월, Q2: 04-06월, Q3: 07-09월, Q4: 10-12월
- 예: strftime('%Y-%m', DOC_DATE) BETWEEN '2024-10' AND '2024-12'
- 데이터 범위: 2023-01-01 ~ 2025-12-31
{period_hint}{region_hint}

**JOIN 예제 (올바른 컬럼 참조)**:

1. 물류비 현황 조회:
```sql
SELECT sh.SUBSIDIARY_ID, cd.COST_TYPE, SUM(cd.COST_AMOUNT) as total_cost
FROM TBL_TX_COST_DETAIL cd
JOIN TBL_TX_SALES_HEADER sh ON cd.ORDER_NO = sh.ORDER_NO
WHERE cd.COST_TYPE = 'LOG'
  AND sh.SUBSIDIARY_ID IN ('LGEUS', 'LGECA')
  AND sh.DOC_DATE BETWEEN '2025-10-01' AND '2025-12-31'
GROUP BY sh.SUBSIDIARY_ID, cd.COST_TYPE
```

2. 매출 및 가격조건 조회:
```sql
SELECT sh.SUBSIDIARY_ID, pc.COND_TYPE, SUM(pc.COND_VALUE) as total_amount
FROM TBL_TX_PRICE_CONDITION pc
JOIN TBL_TX_SALES_HEADER sh ON pc.ORDER_NO = sh.ORDER_NO
WHERE sh.DOC_DATE BETWEEN '2025-10-01' AND '2025-12-31'
GROUP BY sh.SUBSIDIARY_ID, pc.COND_TYPE
```

**주의**:
- TBL_TX_SALES_HEADER에는 ORDER_NO, DOC_DATE, SUBSIDIARY_ID, TOTAL_NET_VALUE만 있음 (ITEM_NO 없음!)
- ITEM_NO는 TBL_TX_SALES_ITEM과 TBL_TX_COST_DETAIL에만 있음

**분석 가이드라인**:
1. 수익성 분석 시 전년 동기 비교 우선 (YoY)
2. COST_TYPE별 분석으로 원가 상승 요인 파악
3. COND_TYPE별 분석으로 매출 감소 요인 파악
4. 변동률 계산: (New - Old) / Old * 100

=== 사용자 질문 ===
{question}

=== 태스크 ===
위 질문에 답하기 위한 SQLite 쿼리를 생성하세요.
SQL 쿼리만 반환하세요. 설명이나 마크다운 없이 순수 SQL만 작성하세요.
"""

    def _create_reasoning_prompt(self, question: str) -> str:
        """추론 과정 생성 프롬프트"""
        return f"""당신은 LG전자 HE사업부 데이터 분석 전문가입니다.

{self.schema_info}

=== 비즈니스 컨텍스트 ===
- 북미(NA): SUBSIDIARY_ID IN ('LGEUS', 'LGECA')
- 유럽(EU): SUBSIDIARY_ID IN ('LGEUK', 'LGEDE', 'LGEFR')
- 한국(KR): SUBSIDIARY_ID = 'LGEKR'
- 데이터 범위: 2023-01-01 ~ 2025-12-31

=== 사용자 질문 ===
{question}

=== 태스크 ===
SQL 쿼리 생성 전, 분석 전략을 설명하세요:

1. **질문 해석**: 사용자가 무엇을 요청하는가?
2. **지역 매핑**: 언급된 지역의 SUBSIDIARY_ID는?
3. **기간 선택**: 비교할 기간은? (YoY 우선)
4. **분석 방법**: 어떤 테이블과 메트릭이 필요한가?

한국어로 간결하게 3-5줄로 작성하세요.
"""

    def generate_reasoning(self, question: str) -> str:
        """추론 과정 생성"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst."},
                    {"role": "user", "content": self._create_reasoning_prompt(question)}
                ],
                temperature=0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"추론 생성 오류: {e}"

    def generate(self, question: str, context: Dict = None, with_reasoning: bool = False) -> SQLGenerationResult:
        """
        SQL 쿼리 생성

        Args:
            question: 자연어 질문
            context: 추가 컨텍스트 (period, region 등)
            with_reasoning: Chain-of-Thought 추론 포함 여부

        Returns:
            SQLGenerationResult
        """
        reasoning = ""

        try:
            # Chain-of-Thought 추론 (선택적)
            if with_reasoning:
                reasoning = self.generate_reasoning(question)

            # SQL 생성
            prompt = self._create_sql_prompt(question, context)

            response = self.client.chat.completions.create(
                model="gpt-4o",  # 더 정확한 SQL 생성을 위해 gpt-4o 사용
                messages=[
                    {"role": "system", "content": "You are an expert SQL query generator. Return only valid SQLite queries without any markdown or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )

            sql_query = response.choices[0].message.content.strip()

            # SQL 코드 블록 제거
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0]
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0]

            sql_query = sql_query.strip()

            return SQLGenerationResult(
                query=sql_query,
                reasoning=reasoning,
                success=True
            )

        except Exception as e:
            return SQLGenerationResult(
                query="",
                reasoning=reasoning,
                success=False,
                error=str(e)
            )

    def execute(self, question: str, context: Dict = None) -> ToolResult:
        """Tool 인터페이스 구현"""
        result = self.generate(question, context)

        if result.success:
            return ToolResult(
                success=True,
                data={
                    "query": result.query,
                    "reasoning": result.reasoning
                }
            )
        else:
            return ToolResult(
                success=False,
                error=result.error
            )
