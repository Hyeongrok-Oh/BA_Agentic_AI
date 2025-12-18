"""
Hypothesis Validator Agent - 가설 검증 에이전트

역할:
- SQLGenerator를 사용하여 각 가설을 ERP 데이터로 검증
- 임계값 이상 변동이 있는 가설만 validated로 표시
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..base import BaseAgent, AgentContext
from ..tools import SQLGenerator, SQLExecutor
from .hypothesis_generator import Hypothesis


@dataclass
class ValidationResult:
    """검증 결과"""
    hypothesis_id: str
    validated: bool
    change_percent: float
    previous_value: float
    current_value: float
    direction: str
    details: str
    sql_query: str = ""  # 사용된 SQL 쿼리


class HypothesisValidator(BaseAgent):
    """가설 검증 에이전트"""

    name = "hypothesis_validator"
    description = "SQLGenerator를 사용하여 가설을 ERP 데이터로 검증합니다."

    def __init__(self, api_key: str = None, db_path: str = None):
        super().__init__(api_key)
        self.sql_generator = SQLGenerator(db_path, api_key)
        self.sql_executor = SQLExecutor(db_path)
        self.add_tool(self.sql_generator)
        self.add_tool(self.sql_executor)

    def validate(
        self,
        hypotheses: List[Hypothesis],
        period: Dict,
        region: str = None,
        threshold: float = 5.0
    ) -> List[Hypothesis]:
        """
        가설 목록 검증

        Args:
            hypotheses: 검증할 가설 목록
            period: {"year": 2024, "quarter": 4}
            region: 지역 코드
            threshold: 변동 임계값 (%)

        Returns:
            검증된 가설 목록
        """
        year = period.get("year", 2024)
        quarter = period.get("quarter", 4)

        # 현재 기간 vs 전년 동기
        curr_start, curr_end = self._get_quarter_range(year, quarter)
        prev_start, prev_end = self._get_quarter_range(year - 1, quarter)

        validated_hypotheses = []

        for hypothesis in hypotheses:
            result = self._validate_single(
                hypothesis,
                prev_start, prev_end,
                curr_start, curr_end,
                region,  # 지역 코드 직접 전달
                threshold
            )

            if result:
                hypothesis.validated = result.validated
                hypothesis.validation_data = {
                    "change_percent": result.change_percent,
                    "previous_value": result.previous_value,
                    "current_value": result.current_value,
                    "direction": result.direction,
                    "details": result.details,
                    "sql_query": result.sql_query  # SQL 쿼리 저장
                }

                if result.validated:
                    validated_hypotheses.append(hypothesis)

        return validated_hypotheses

    def _validate_single(
        self,
        hypothesis: Hypothesis,
        prev_start: str, prev_end: str,
        curr_start: str, curr_end: str,
        region: str,
        threshold: float
    ) -> Optional[ValidationResult]:
        """단일 가설 검증 (SQLGenerator 사용)"""

        # 가설 검증용 자연어 질문 생성
        region_text = self._get_region_text(region)
        question = self._build_validation_question(
            hypothesis.factor,
            prev_start, prev_end,
            curr_start, curr_end,
            region_text
        )

        sql_query = ""  # SQL 쿼리 저장용

        try:
            # 1. SQLGenerator로 쿼리 생성
            context = {
                "period": {"prev_start": prev_start, "curr_start": curr_start},
                "region": region
            }
            gen_result = self.sql_generator.generate(question, context, with_reasoning=False)

            if not gen_result.success:
                print(f"SQL 생성 실패 ({hypothesis.id}): {gen_result.error}")
                return None

            sql_query = gen_result.query  # SQL 쿼리 저장

            # 2. SQLExecutor로 실행
            exec_result = self.sql_executor.execute(gen_result.query)

            if not exec_result.success or exec_result.data is None:
                print(f"SQL 실행 실패 ({hypothesis.id}): {exec_result.error}")
                return None

            data = exec_result.data.to_dict('records')

            if not data:
                return None

            # Previous와 Current 값 추출
            prev_row = next((r for r in data if r.get('PERIOD') == 'Previous'), None)
            curr_row = next((r for r in data if r.get('PERIOD') == 'Current'), None)

            if not prev_row or not curr_row:
                return None

            # 첫 번째 숫자 값 사용
            value_key = [k for k in prev_row.keys() if k != 'PERIOD'][0]

            prev_value = float(prev_row[value_key] or 0)
            curr_value = float(curr_row[value_key] or 0)

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
                details=f"{hypothesis.factor}: {prev_value:,.0f} → {curr_value:,.0f} ({change_percent:+.1f}%)",
                sql_query=sql_query
            )

        except Exception as e:
            print(f"검증 오류 ({hypothesis.id}): {e}")
            return None

    def _build_validation_question(
        self,
        factor: str,
        prev_start: str, prev_end: str,
        curr_start: str, curr_end: str,
        region_text: str
    ) -> str:
        """가설 검증용 자연어 질문 생성"""
        return f"""{region_text} {factor} 비교 분석:
- Previous 기간: {prev_start} ~ {prev_end}
- Current 기간: {curr_start} ~ {curr_end}

두 기간의 {factor} 총합을 비교하여 PERIOD 컬럼으로 'Previous'와 'Current'로 구분해서 보여줘.
결과는 반드시 PERIOD, TOTAL_VALUE 컬럼을 포함해야 해."""

    def _get_region_text(self, region: str) -> str:
        """지역 코드를 텍스트로 변환"""
        if not region:
            return "전체 지역"

        region_map = {
            "na": "북미 지역",
            "eu": "유럽 지역",
            "kr": "한국 지역"
        }
        return region_map.get(region.lower(), region)

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

    def run(self, context: AgentContext) -> Dict[str, Any]:
        """Agent 실행"""
        hypotheses = context.metadata.get("hypotheses", [])
        period = context.metadata.get("period", {"year": 2024, "quarter": 4})
        region = context.metadata.get("region")
        threshold = context.metadata.get("threshold", 5.0)

        validated = self.validate(
            hypotheses=hypotheses,
            period=period,
            region=region,
            threshold=threshold
        )

        result = {
            "validated_hypotheses": validated,
            "validated_count": len(validated),
            "total_count": len(hypotheses)
        }

        context.add_step("hypothesis_validation", {
            "validated_count": len(validated),
            "total_count": len(hypotheses)
        })

        return result
