"""
가설 검증기 - SQL Agent를 사용하여 가설을 ERP 데이터로 검증
"""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

# SQL Agent import
sys.path.insert(0, '/Users/hyeongrokoh/BI/sql')
from sql_agent import SQLAgent

from .hypothesis_generator import Hypothesis


# 가설 검증용 SQL 템플릿
VALIDATION_QUERIES = {
    # 원가 관련
    "cost_mat": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            ROUND(SUM(COST_AMOUNT * ORDER_QTY), 2) as TOTAL_COST,
            ROUND(AVG(COST_AMOUNT), 2) as AVG_UNIT_COST
        FROM TBL_TX_COST_DETAIL CD
        JOIN TBL_TX_SALES_ITEM SI ON CD.ORDER_NO = SI.ORDER_NO AND CD.ITEM_NO = SI.ITEM_NO
        JOIN TBL_TX_SALES_HEADER SH ON SI.ORDER_NO = SH.ORDER_NO
        WHERE CD.COST_TYPE = 'MAT'
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """,

    "cost_log": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            ROUND(SUM(COST_AMOUNT * ORDER_QTY), 2) as TOTAL_COST,
            ROUND(AVG(COST_AMOUNT), 2) as AVG_UNIT_COST
        FROM TBL_TX_COST_DETAIL CD
        JOIN TBL_TX_SALES_ITEM SI ON CD.ORDER_NO = SI.ORDER_NO AND CD.ITEM_NO = SI.ITEM_NO
        JOIN TBL_TX_SALES_HEADER SH ON SI.ORDER_NO = SH.ORDER_NO
        WHERE CD.COST_TYPE = 'LOG'
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """,

    "cost_tar": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            ROUND(SUM(COST_AMOUNT * ORDER_QTY), 2) as TOTAL_COST,
            ROUND(AVG(COST_AMOUNT), 2) as AVG_UNIT_COST
        FROM TBL_TX_COST_DETAIL CD
        JOIN TBL_TX_SALES_ITEM SI ON CD.ORDER_NO = SI.ORDER_NO AND CD.ITEM_NO = SI.ITEM_NO
        JOIN TBL_TX_SALES_HEADER SH ON SI.ORDER_NO = SH.ORDER_NO
        WHERE CD.COST_TYPE = 'TAR'
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """,

    # 가격 조건 관련
    "pricing_pp": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            ROUND(SUM(ABS(PC.COND_VALUE)), 2) as TOTAL_PP,
            COUNT(*) as COUNT
        FROM TBL_TX_PRICE_CONDITION PC
        JOIN TBL_TX_SALES_HEADER SH ON PC.ORDER_NO = SH.ORDER_NO
        WHERE PC.COND_TYPE = 'ZPRO'
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """,

    "pricing_discount": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            ROUND(SUM(ABS(PC.COND_VALUE)), 2) as TOTAL_DISCOUNT,
            COUNT(*) as COUNT
        FROM TBL_TX_PRICE_CONDITION PC
        JOIN TBL_TX_SALES_HEADER SH ON PC.ORDER_NO = SH.ORDER_NO
        WHERE PC.COND_TYPE = 'K007'
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """,

    "pricing_mdf": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            ROUND(SUM(ABS(PC.COND_VALUE)), 2) as TOTAL_MDF,
            COUNT(*) as COUNT
        FROM TBL_TX_PRICE_CONDITION PC
        JOIN TBL_TX_SALES_HEADER SH ON PC.ORDER_NO = SH.ORDER_NO
        WHERE PC.COND_TYPE = 'ZMDF'
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """,

    # 매출 관련
    "revenue_total": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            ROUND(SUM(TOTAL_NET_VALUE), 2) as TOTAL_REVENUE,
            COUNT(*) as ORDER_COUNT
        FROM TBL_TX_SALES_HEADER
        WHERE 1=1
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """,

    "volume_total": """
        SELECT
            CASE WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}' THEN 'Previous'
                 WHEN strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}' THEN 'Current' END as PERIOD,
            SUM(ORDER_QTY) as TOTAL_QUANTITY,
            ROUND(AVG(ORDER_QTY), 2) as AVG_QTY_PER_ORDER
        FROM TBL_TX_SALES_ITEM SI
        JOIN TBL_TX_SALES_HEADER SH ON SI.ORDER_NO = SH.ORDER_NO
        WHERE 1=1
        {region_filter}
        AND (strftime('%Y-%m', DOC_DATE) BETWEEN '{prev_start}' AND '{prev_end}'
             OR strftime('%Y-%m', DOC_DATE) BETWEEN '{curr_start}' AND '{curr_end}')
        GROUP BY PERIOD
        ORDER BY PERIOD
    """
}

# Factor to Query 매핑
FACTOR_TO_QUERY = {
    "재료비": "cost_mat",
    "Material Cost": "cost_mat",
    "MAT": "cost_mat",
    "물류비": "cost_log",
    "Logistics Cost": "cost_log",
    "LOG": "cost_log",
    "관세": "cost_tar",
    "Tariff": "cost_tar",
    "TAR": "cost_tar",
    "Price Protection": "pricing_pp",
    "PP": "pricing_pp",
    "프로모션": "pricing_pp",
    "할인": "pricing_discount",
    "Discount": "pricing_discount",
    "MDF": "pricing_mdf",
    "매출": "revenue_total",
    "Revenue": "revenue_total",
    "판매량": "volume_total",
    "Volume": "volume_total",
    "수량": "volume_total"
}


@dataclass
class ValidationResult:
    """검증 결과"""
    hypothesis_id: str
    validated: bool
    change_percent: float
    previous_value: float
    current_value: float
    direction: str  # increased, decreased, stable
    details: str


class HypothesisValidator:
    """SQL Agent를 사용한 가설 검증기"""

    def __init__(
        self,
        db_path: str = "/Users/hyeongrokoh/BI/sql/lge_he_erp.db",
        api_key: Optional[str] = None
    ):
        self.db_path = db_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sql_agent = SQLAgent(db_path, self.api_key)

    def validate_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        period: Dict,
        region: str = None,
        threshold: float = 5.0  # 변동 임계값 (%)
    ) -> List[Hypothesis]:
        """가설 목록 검증"""

        # 기간 파싱
        year = period.get("year", 2024)
        quarter = period.get("quarter", 4)

        # 현재 기간 vs 전년 동기
        curr_start, curr_end = self._get_quarter_range(year, quarter)
        prev_start, prev_end = self._get_quarter_range(year - 1, quarter)

        # 지역 필터
        region_filter = self._get_region_filter(region)

        validated_hypotheses = []

        for hypothesis in hypotheses:
            result = self._validate_single(
                hypothesis,
                prev_start, prev_end,
                curr_start, curr_end,
                region_filter,
                threshold
            )

            if result:
                hypothesis.validated = result.validated
                hypothesis.validation_data = {
                    "change_percent": result.change_percent,
                    "previous_value": result.previous_value,
                    "current_value": result.current_value,
                    "direction": result.direction,
                    "details": result.details
                }

                if result.validated:
                    validated_hypotheses.append(hypothesis)

        return validated_hypotheses

    def _validate_single(
        self,
        hypothesis: Hypothesis,
        prev_start: str,
        prev_end: str,
        curr_start: str,
        curr_end: str,
        region_filter: str,
        threshold: float
    ) -> Optional[ValidationResult]:
        """단일 가설 검증"""

        # Factor에 맞는 쿼리 찾기
        query_key = None
        for factor_name, qkey in FACTOR_TO_QUERY.items():
            if factor_name.lower() in hypothesis.factor.lower():
                query_key = qkey
                break

        if not query_key:
            # 기본적으로 SQL Agent에게 맡김
            return self._validate_with_agent(hypothesis, prev_start, prev_end, curr_start, curr_end, region_filter)

        # 템플릿 쿼리 실행
        query_template = VALIDATION_QUERIES.get(query_key)
        if not query_template:
            return None

        query = query_template.format(
            prev_start=prev_start,
            prev_end=prev_end,
            curr_start=curr_start,
            curr_end=curr_end,
            region_filter=region_filter
        )

        try:
            df = self.sql_agent.execute_sql(query)

            if df.empty or len(df) < 2:
                return None

            # Previous와 Current 값 추출
            prev_row = df[df['PERIOD'] == 'Previous']
            curr_row = df[df['PERIOD'] == 'Current']

            if prev_row.empty or curr_row.empty:
                return None

            # 첫 번째 숫자 컬럼 사용
            value_col = df.columns[1]  # TOTAL_COST, TOTAL_PP, etc.

            prev_value = float(prev_row[value_col].iloc[0])
            curr_value = float(curr_row[value_col].iloc[0])

            if prev_value == 0:
                change_percent = 100.0 if curr_value > 0 else 0.0
            else:
                change_percent = ((curr_value - prev_value) / prev_value) * 100

            # 방향 판단
            if change_percent > threshold:
                direction = "increased"
            elif change_percent < -threshold:
                direction = "decreased"
            else:
                direction = "stable"

            # 가설과 실제 방향 비교
            hypothesis_direction = hypothesis.direction.lower()
            validated = False

            if hypothesis_direction == "increase" and direction == "increased":
                validated = True
            elif hypothesis_direction == "decrease" and direction == "decreased":
                validated = True

            return ValidationResult(
                hypothesis_id=hypothesis.id,
                validated=validated,
                change_percent=round(change_percent, 1),
                previous_value=prev_value,
                current_value=curr_value,
                direction=direction,
                details=f"{hypothesis.factor}: {prev_value:,.0f} → {curr_value:,.0f} ({change_percent:+.1f}%)"
            )

        except Exception as e:
            print(f"검증 오류 ({hypothesis.id}): {e}")
            return None

    def _validate_with_agent(
        self,
        hypothesis: Hypothesis,
        prev_start: str,
        prev_end: str,
        curr_start: str,
        curr_end: str,
        region_filter: str
    ) -> Optional[ValidationResult]:
        """SQL Agent를 사용한 동적 검증"""
        query = f"""
        {hypothesis.factor}의 {prev_start}~{prev_end} 기간과 {curr_start}~{curr_end} 기간을 비교해줘.
        변화량과 변화율을 계산해줘.
        """

        result = self.sql_agent.query(query)

        if result.get('error') or result.get('data') is None:
            return None

        # 결과 해석은 복잡하므로 일단 None 반환
        return None

    def _get_quarter_range(self, year: int, quarter: int) -> tuple:
        """분기 시작/종료 월 계산"""
        quarter_months = {
            1: ("01", "03"),
            2: ("04", "06"),
            3: ("07", "09"),
            4: ("10", "12")
        }
        start_month, end_month = quarter_months[quarter]
        return f"{year}-{start_month}", f"{year}-{end_month}"

    def _get_region_filter(self, region: str) -> str:
        """지역 필터 SQL 생성"""
        if not region:
            return ""

        region_lower = region.lower()

        if region_lower in ["na", "north america", "북미"]:
            return "AND SH.SUBSIDIARY_ID IN ('LGEUS', 'LGECA')"
        elif region_lower in ["eu", "europe", "유럽"]:
            return "AND SH.SUBSIDIARY_ID IN ('LGEUK', 'LGEDE', 'LGEFR')"
        elif region_lower in ["kr", "korea", "한국"]:
            return "AND SH.SUBSIDIARY_ID = 'LGEKR'"
        else:
            return ""
