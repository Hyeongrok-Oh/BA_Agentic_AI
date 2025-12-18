#!/usr/bin/env python3
"""
Custom SQL Agent - LangChain 없이 순수 Python으로 구현
OpenAI API를 직접 호출하여 SQL 쿼리 생성 및 실행
"""

import os
import sqlite3
import json
import pandas as pd
from typing import Dict, Any, List
from openai import OpenAI


class SQLAgent:
    """LangChain 없는 커스텀 SQL Agent"""

    def __init__(self, db_path: str, api_key: str = None):
        """
        Initialize SQL Agent

        Args:
            db_path: Path to SQLite database
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.db_path = db_path
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.schema_info = self._get_schema_info()

    def _get_schema_info(self) -> str:
        """데이터베이스 스키마 정보 추출"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 모든 테이블 목록 가져오기
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        schema_lines = []
        schema_lines.append("=== DATABASE SCHEMA ===\n")

        for (table_name,) in tables:
            # 테이블 구조 가져오기
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

    def _create_reasoning_prompt(self, user_query: str) -> str:
        """추론 과정 생성을 위한 프롬프트 작성"""
        return f"""You are an expert SQL analyst for LG Electronics Home Entertainment Division.

{self.schema_info}

=== BUSINESS CONTEXT ===

**REGION TO SUBSIDIARY MAPPING (CRITICAL)**:
TBL_ORG_SUBSIDIARY 테이블 구조:
- SUBSIDIARY_ID (PK): 법인 코드
- REGION: 지역 코드 (NA, EU, KR 등)
- CURRENCY: 통화

실제 데이터:
| SUBSIDIARY_ID | REGION | CURRENCY | 설명 |
|---------------|--------|----------|------|
| LGEUS         | NA     | USD      | LG Electronics US |
| LGECA         | NA     | CAD      | LG Electronics Canada |
| LGEKR         | KR     | KRW      | LG Electronics Korea |
| LGEUK         | EU     | GBP      | LG Electronics UK |
| LGEDE         | EU     | EUR      | LG Electronics Germany |
| LGEFR         | EU     | EUR      | LG Electronics France |

**지역별 필터링 방법 (MUST USE)**:
- 북미(North America): WHERE SUBSIDIARY_ID IN ('LGEUS', 'LGECA')
- 유럽(Europe): WHERE SUBSIDIARY_ID IN ('LGEUK', 'LGEDE', 'LGEFR')
- 한국(Korea): WHERE SUBSIDIARY_ID = 'LGEKR'
- 전체: 필터 없음

⚠️ 주의: REGION 컬럼으로 직접 필터링하지 말고, SUBSIDIARY_ID로 필터링할 것!

**PERIOD DEFINITIONS**:
- Database contains data from 2023-01-01 to 2025-12-31 (3 years)
- Q4 2024 = 2024-10-01 to 2024-12-31
- Q3 2024 = 2024-07-01 to 2024-09-30
- Q4 2023 = 2023-10-01 to 2023-12-31
- Q3 2023 = 2023-07-01 to 2023-09-30

=== USER QUESTION ===
{user_query}

=== TASK ===
Before generating SQL, explain your analysis strategy step-by-step:

1. **Question Interpretation**: What is the user asking for?
2. **Region Mapping**: If a region is mentioned (e.g., "북미", "North America"), which SUBSIDIARY_IDs should be used?
3. **Time Period Selection**:
   - For profit decline analysis, which periods should be compared?
   - Priority: Year-over-year (2023 Q4 vs 2024 Q4) > Quarter-over-quarter (Q3 vs Q4)
   - The database contains data from 2023-01-01 to 2025-12-31
   - ALWAYS prefer year-over-year comparison when analyzing 2024 Q4 (compare with 2023 Q4)
4. **Analysis Approach**: Which tables and metrics are needed?

Provide a clear, concise reasoning in Korean (3-5 bullet points).
"""

    def _create_sql_prompt(self, user_query: str) -> str:
        """SQL 쿼리 생성을 위한 프롬프트 작성"""
        return f"""You are an expert SQL analyst for LG Electronics Home Entertainment Division.

{self.schema_info}

=== BUSINESS LOGIC ===

**PRICING CONDITIONS (TBL_TX_PRICE_CONDITION)**:
- 'ZPR0' = Gross Price (Positive revenue)
- 'ZPRO' = Price Protection (Negative - deduction/promotional cost)
- 'K007' = Volume Discount (Negative)
- 'ZMDF' = Marketing Development Fund (Negative)

**COST TYPES (TBL_TX_COST_DETAIL)**:
- 'MAT' = Material Cost
- 'LOG' = Logistics Cost
- 'TAR' = Tariff and customs duties
- 'OH' = Manufacturing overhead

**REGION TO SUBSIDIARY MAPPING (CRITICAL - MUST FOLLOW)**:
TBL_ORG_SUBSIDIARY 테이블:
| SUBSIDIARY_ID | REGION | CURRENCY | 설명 |
|---------------|--------|----------|------|
| LGEUS         | NA     | USD      | LG Electronics US |
| LGECA         | NA     | CAD      | LG Electronics Canada |
| LGEKR         | KR     | KRW      | LG Electronics Korea |
| LGEUK         | EU     | GBP      | LG Electronics UK |
| LGEDE         | EU     | EUR      | LG Electronics Germany |
| LGEFR         | EU     | EUR      | LG Electronics France |

**지역별 SQL WHERE 조건 (반드시 이 패턴 사용)**:
- 북미(North America, 미국, 캐나다): SUBSIDIARY_ID IN ('LGEUS', 'LGECA')
- 유럽(Europe, EU): SUBSIDIARY_ID IN ('LGEUK', 'LGEDE', 'LGEFR')
- 한국(Korea, KR): SUBSIDIARY_ID = 'LGEKR'
- 전체(Global): 필터 없음

⚠️ 절대 REGION 컬럼으로 직접 필터링하지 말 것! 반드시 SUBSIDIARY_ID 사용!

**PERIOD DEFINITIONS**:
- Database: 2023-01-01 ~ 2025-12-31 (3년)
- Q1 = 01-01 ~ 03-31, Q2 = 04-01 ~ 06-30, Q3 = 07-01 ~ 09-30, Q4 = 10-01 ~ 12-31
- 분기 필터: strftime('%Y-%m', DOC_DATE) BETWEEN 'YYYY-MM' AND 'YYYY-MM'

**ANALYSIS GUIDELINES**:
When user asks about "profit decline", "margin drop", or "root cause analysis":
1. ALWAYS prefer year-over-year comparison (e.g., 2023 Q4 vs 2024 Q4) over quarter-over-quarter
2. Break down by COST_TYPE to identify cost spikes (MAT, LOG, TAR, OH)
3. Break down by COND_TYPE to identify revenue issues (ZPRO, K007, ZMDF)
4. Include percentage calculations: (New - Old) / Old * 100
5. Use GROUP BY for time periods and cost/condition types

Example pattern for year-over-year Q4 comparison:
```sql
SELECT
    CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '2023-10' AND '2023-12' THEN 'Q4 2023'
         WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '2024-10' AND '2024-12' THEN 'Q4 2024' END as QUARTER,
    COST_TYPE,
    ROUND(AVG(COST_AMOUNT), 2) as AVG_COST,
    ROUND(SUM(COST_AMOUNT * ORDER_QTY), 2) as TOTAL_COST
FROM ...
GROUP BY QUARTER, COST_TYPE
ORDER BY QUARTER, TOTAL_COST DESC
```

=== USER QUESTION ===
{user_query}

=== INSTRUCTIONS ===
Generate a SQLite query to answer the user's question.
Return ONLY the SQL query, nothing else.
The query must be valid SQLite syntax.
Use appropriate JOINs based on the schema structure.

IMPORTANT: If the question asks about profit decline or root causes, generate a query that:
- Compares year-over-year periods (2023 Q4 vs 2024 Q4) NOT quarter-over-quarter
- Shows breakdown by cost types OR condition types
- Includes both totals and averages per unit
- Uses strftime('%Y-%m', DOC_DATE) for year-specific filtering
"""

    def _generate_reasoning(self, user_query: str) -> str:
        """분석 전략 및 추론 과정 생성"""
        prompt = self._create_reasoning_prompt(user_query)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert financial analyst explaining your analysis strategy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )

        reasoning = response.choices[0].message.content.strip()
        return reasoning

    def generate_sql(self, user_query: str) -> str:
        """LLM을 사용하여 SQL 쿼리 생성"""
        prompt = self._create_sql_prompt(user_query)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert SQL query generator. Return only valid SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )

        sql_query = response.choices[0].message.content.strip()

        # SQL 코드 블록 제거 (```sql ... ``` 형식)
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sql_query
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        return sql_query

    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """SQL 쿼리 실행 및 결과 반환"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(sql_query, conn)
            return df
        finally:
            conn.close()

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        사용자 질문을 받아서 SQL 생성 및 실행

        Returns:
            {
                'question': 사용자 질문,
                'reasoning': Agent의 판단 근거,
                'sql': 생성된 SQL 쿼리,
                'data': 실행 결과 DataFrame,
                'error': 에러 메시지 (있을 경우)
            }
        """
        result = {
            'question': user_query,
            'reasoning': None,
            'sql': None,
            'data': None,
            'error': None
        }

        try:
            # 1. 추론 과정 생성 (Chain-of-Thought)
            reasoning = self._generate_reasoning(user_query)
            result['reasoning'] = reasoning

            # 2. SQL 쿼리 생성
            sql_query = self.generate_sql(user_query)
            result['sql'] = sql_query

            # 3. SQL 쿼리 실행
            df = self.execute_sql(sql_query)
            result['data'] = df

        except Exception as e:
            result['error'] = str(e)

        return result
