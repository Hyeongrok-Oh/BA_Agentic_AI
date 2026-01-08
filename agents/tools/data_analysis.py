"""
Data Analysis Module - Shapley Value 기반 KPI 기여도 분석

주요 기능:
1. 시계열 데이터 수집 (Driver, KPI)
2. 회귀모델 학습 (Ridge)
3. SHAP 기여도 계산
4. LLM 기반 해석 생성
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import json

from ..base import BaseTool, ToolResult


# =============================================================================
# Driver-Table 매핑 (hypothesis_validator.py와 동일)
# =============================================================================

DRIVER_CONFIG = {
    # T1: ERP 직접 검증
    "출하량": {"table": "TR_SALES", "column": "QTY", "date_col": "SALES_DATE"},
    "판매량": {"table": "TR_SALES", "column": "QTY", "date_col": "SALES_DATE"},
    "OLED비중": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE", "filter": "prod.DISPLAY_TYPE = 'OLED'"},
    "프리미엄비중": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE", "filter": "prod.IS_PREMIUM = 'Y'"},
    "TV평균판매가": {"table": "TR_SALES", "column": "REVENUE_USD/QTY", "date_col": "SALES_DATE"},
    "할인율": {"table": "TR_EXPENSE", "column": "PROMOTION_COST", "date_col": "EXPENSE_DATE"},
    "프로모션비용": {"table": "TR_EXPENSE", "column": "PROMOTION_COST", "date_col": "EXPENSE_DATE"},
    "패널원가": {"table": "TR_PURCHASE", "column": "PANEL_PRICE_USD", "date_col": "PURCHASE_DATE"},
    "제조원가": {"table": "TR_PURCHASE", "column": "TOTAL_COGS_USD", "date_col": "PURCHASE_DATE"},
    "물류비": {"table": "TR_EXPENSE", "column": "LOGISTICS_COST", "date_col": "EXPENSE_DATE"},
    "원재료비": {"table": "TR_PURCHASE", "column": "RAW_MATERIAL_INDEX", "date_col": "PURCHASE_DATE"},
    "달러환율": {"table": "EXT_MACRO", "column": "EXCHANGE_RATE_KRW_USD", "date_col": "DATA_DATE"},
    # T2: Proxy
    "글로벌TV수요": {"table": "EXT_MARKET", "column": "TOTAL_SHIPMENT_10K", "date_col": "DATA_DATE"},
    "소비심리": {"table": "EXT_MACRO", "column": "CSI_INDEX", "date_col": "DATA_DATE"},
    "인플레이션": {"table": "EXT_MACRO", "column": "INFLATION_RATE", "date_col": "DATA_DATE"},
    "금리": {"table": "EXT_MACRO", "column": "INTEREST_RATE", "date_col": "DATA_DATE"},
}

KPI_CONFIG = {
    "revenue": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE"},
    "profit": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE"},  # 간략화
    "quantity": {"table": "TR_SALES", "column": "QTY", "date_col": "SALES_DATE"},
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DriverContribution:
    """Driver 기여도 결과"""
    driver_id: str
    driver_name: str
    shapley_value: float
    contribution_pct: float
    direction: str  # "positive" | "negative"
    rank: int
    interpretation: str = ""


@dataclass
class HypothesisValidation:
    """가설 검증 결과"""
    hypothesis_id: str
    driver_id: str
    description: str
    expected_direction: str
    validation_status: str  # "validated" | "partially_validated" | "not_validated"
    confidence_score: float
    reasoning: str


@dataclass
class AnalysisResult:
    """전체 분석 결과"""
    kpi_change_summary: str
    kpi_change_pct: float
    top_drivers: List[DriverContribution]
    hypotheses: List[HypothesisValidation]
    final_explanation: str
    model_r_squared: float
    data_quality: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Main Class
# =============================================================================

class DataAnalyzer(BaseTool):
    """
    KPI 기여도 분석기 (Shapley Value 기반)

    사용법:
        analyzer = DataAnalyzer(sql_executor)
        result = analyzer.analyze(hypotheses, kpi_id="revenue", period={"year": 2024, "quarter": 4})
    """

    name = "data_analyzer"
    description = "Shapley Value 기반 KPI 기여도 분석"

    # LLM Prompt for interpretation
    VALIDATION_PROMPT = """You are the Hypothesis Validation Agent.

INPUTS:
- KPI Change: {kpi_change_pct:.1f}%
- SHAP Contributions:
{contributions_text}

- Hypotheses to validate:
{hypotheses_text}

ANALYSIS STEPS:
1. Rank drivers by absolute contribution
2. For each hypothesis:
   - Check if driver's rank <= 3 OR contribution_pct >= 10%
   - Compare expected vs actual direction
   - Assign validation_status: "validated", "partially_validated", or "not_validated"
3. Compute confidence_score (0-1)

OUTPUT JSON:
{{
  "kpi_change_summary": "<text>",
  "top_drivers": [
    {{"driver_id": "<id>", "direction": "positive/negative", "contribution_pct": <float>, "rank": <int>, "interpretation": "<text>"}}
  ],
  "hypotheses": [
    {{"hypothesis_id": "<id>", "validation_status": "<status>", "confidence_score": <float>, "reasoning": "<text>"}}
  ],
  "final_explanation": "<summary paragraph>"
}}"""

    def __init__(self, sql_executor, llm_client=None):
        """
        Args:
            sql_executor: SQLExecutor 인스턴스
            llm_client: OpenAI client (optional, for LLM interpretation)
        """
        self.sql_executor = sql_executor
        self.llm_client = llm_client
        self._model_cache: Dict[str, Any] = {}
        self._series_cache: Dict[str, pd.Series] = {}

    # =========================================================================
    # 1. 시계열 데이터 수집
    # =========================================================================

    def get_driver_time_series(
        self,
        driver_id: str,
        months: int = 24
    ) -> Optional[pd.Series]:
        """
        Driver 월별 시계열 추출

        Args:
            driver_id: Driver 식별자 (한글)
            months: 수집할 개월 수

        Returns:
            월별 집계된 pd.Series (index: YYYY-MM)
        """
        cache_key = f"driver_{driver_id}_{months}"
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        config = DRIVER_CONFIG.get(driver_id)
        if not config:
            print(f"[DataAnalyzer] Unknown driver: {driver_id}")
            return None

        table = config["table"]
        column = config["column"]
        date_col = config["date_col"]
        filter_cond = config.get("filter", "")

        # ASP 계산인 경우 처리
        if "/" in column:
            num, denom = column.split("/")
            value_expr = f"SUM({num}) * 1.0 / NULLIF(SUM({denom}), 0)"
        else:
            value_expr = f"SUM({column})"

        # Filter JOIN (if needed)
        join_clause = ""
        if filter_cond and "prod." in filter_cond:
            join_clause = "LEFT JOIN MD_PRODUCT prod ON t.PRODUCT_ID = prod.PRODUCT_ID"

        where_filter = f"AND {filter_cond}" if filter_cond else ""

        sql = f"""
        SELECT
            strftime('%Y-%m', {date_col}) as period,
            {value_expr} as value
        FROM {table} t
        {join_clause}
        WHERE {date_col} >= date('now', '-{months} months')
        {where_filter}
        GROUP BY strftime('%Y-%m', {date_col})
        ORDER BY period
        """

        result = self.sql_executor.execute(sql)
        if not result.success or result.data is None or len(result.data) == 0:
            print(f"[DataAnalyzer] No data for driver: {driver_id}")
            return None

        series = pd.Series(
            result.data['value'].values,
            index=pd.Index(result.data['period'].values, name='period'),
            name=driver_id
        )

        self._series_cache[cache_key] = series
        return series

    def get_kpi_time_series(
        self,
        kpi_id: str = "revenue",
        months: int = 24
    ) -> Optional[pd.Series]:
        """
        KPI 월별 시계열 추출

        Args:
            kpi_id: KPI 식별자 ("revenue", "profit", "quantity")
            months: 수집할 개월 수

        Returns:
            월별 집계된 pd.Series
        """
        cache_key = f"kpi_{kpi_id}_{months}"
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        config = KPI_CONFIG.get(kpi_id)
        if not config:
            print(f"[DataAnalyzer] Unknown KPI: {kpi_id}")
            return None

        sql = f"""
        SELECT
            strftime('%Y-%m', {config['date_col']}) as period,
            SUM({config['column']}) as value
        FROM {config['table']}
        WHERE {config['date_col']} >= date('now', '-{months} months')
        GROUP BY strftime('%Y-%m', {config['date_col']})
        ORDER BY period
        """

        result = self.sql_executor.execute(sql)
        if not result.success or result.data is None or len(result.data) == 0:
            return None

        series = pd.Series(
            result.data['value'].values,
            index=pd.Index(result.data['period'].values, name='period'),
            name=kpi_id
        )

        self._series_cache[cache_key] = series
        return series

    def collect_all_series(
        self,
        driver_ids: List[str],
        kpi_id: str = "revenue",
        months: int = 24
    ) -> Tuple[Dict[str, pd.Series], pd.Series, Dict[str, Any]]:
        """
        모든 Driver + KPI 시계열 수집

        Returns:
            (driver_series_dict, kpi_series, data_quality_info)
        """
        driver_series = {}
        quality_info = {
            "total_drivers": len(driver_ids),
            "valid_drivers": 0,
            "missing_drivers": [],
            "min_periods": 24,
            "actual_periods": 0
        }

        for driver_id in driver_ids:
            series = self.get_driver_time_series(driver_id, months)
            if series is not None and len(series) >= 12:  # 최소 12개월
                driver_series[driver_id] = series
                quality_info["valid_drivers"] += 1
                quality_info["min_periods"] = min(quality_info["min_periods"], len(series))
            else:
                quality_info["missing_drivers"].append(driver_id)

        kpi_series = self.get_kpi_time_series(kpi_id, months)
        if kpi_series is not None:
            quality_info["actual_periods"] = len(kpi_series)

        return driver_series, kpi_series, quality_info

    # =========================================================================
    # 2. 회귀모델 학습
    # =========================================================================

    def build_model(
        self,
        driver_series: Dict[str, pd.Series],
        kpi_series: pd.Series
    ) -> Tuple[Any, Any, List[str], float]:
        """
        Ridge 회귀모델 학습

        Args:
            driver_series: Driver별 시계열 dict
            kpi_series: KPI 시계열

        Returns:
            (model, scaler, driver_names, r_squared)
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        # 공통 기간만 추출
        common_index = kpi_series.index
        for series in driver_series.values():
            common_index = common_index.intersection(series.index)

        if len(common_index) < 12:
            raise ValueError(f"공통 기간이 부족합니다: {len(common_index)}개월")

        # DataFrame 구성
        X_data = {name: series.loc[common_index].values
                  for name, series in driver_series.items()}
        X = pd.DataFrame(X_data)
        y = kpi_series.loc[common_index].values

        # 결측치 처리 (선형 보간)
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ridge 회귀 (다중공선성 완화)
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        r_squared = model.score(X_scaled, y)

        return model, scaler, list(X.columns), r_squared

    # =========================================================================
    # 3. SHAP 기여도 계산
    # =========================================================================

    def calculate_contributions(
        self,
        model,
        scaler,
        driver_names: List[str],
        driver_series: Dict[str, pd.Series],
        kpi_series: pd.Series
    ) -> List[DriverContribution]:
        """
        SHAP 기반 기여도 계산

        Args:
            model: 학습된 Ridge 모델
            scaler: StandardScaler
            driver_names: Driver 이름 리스트
            driver_series: Driver 시계열 dict
            kpi_series: KPI 시계열

        Returns:
            기여도 순으로 정렬된 DriverContribution 리스트
        """
        try:
            import shap
        except ImportError:
            print("[DataAnalyzer] shap 라이브러리가 설치되지 않았습니다. pip install shap")
            return self._fallback_contributions(model, scaler, driver_names, driver_series)

        # 공통 기간
        common_index = kpi_series.index
        for series in driver_series.values():
            common_index = common_index.intersection(series.index)

        # 현재값 (최근 월)과 기준값 (평균)
        current_values = [driver_series[name].loc[common_index].iloc[-1]
                         for name in driver_names]
        baseline_values = [driver_series[name].loc[common_index].mean()
                          for name in driver_names]

        current_scaled = scaler.transform([current_values])
        baseline_scaled = scaler.transform([baseline_values])

        # SHAP Explainer
        explainer = shap.LinearExplainer(model, baseline_scaled)
        shap_values = explainer.shap_values(current_scaled)[0]

        # 결과 정리
        total_contribution = sum(abs(v) for v in shap_values)
        contributions = []

        for i, driver in enumerate(driver_names):
            shap_val = float(shap_values[i])
            contrib_pct = abs(shap_val) / total_contribution * 100 if total_contribution > 0 else 0

            contributions.append(DriverContribution(
                driver_id=driver,
                driver_name=driver,
                shapley_value=shap_val,
                contribution_pct=round(contrib_pct, 1),
                direction="positive" if shap_val > 0 else "negative",
                rank=0,  # 나중에 설정
                interpretation=""
            ))

        # 순위 부여 (절대값 기준)
        contributions.sort(key=lambda x: abs(x.shapley_value), reverse=True)
        for i, contrib in enumerate(contributions):
            contrib.rank = i + 1

        return contributions

    def _fallback_contributions(
        self,
        model,
        scaler,
        driver_names: List[str],
        driver_series: Dict[str, pd.Series]
    ) -> List[DriverContribution]:
        """SHAP 없이 회귀계수 기반 기여도 계산 (fallback)"""
        coefficients = model.coef_
        total = sum(abs(c) for c in coefficients)

        contributions = []
        for i, driver in enumerate(driver_names):
            coef = float(coefficients[i])
            contrib_pct = abs(coef) / total * 100 if total > 0 else 0

            contributions.append(DriverContribution(
                driver_id=driver,
                driver_name=driver,
                shapley_value=coef,
                contribution_pct=round(contrib_pct, 1),
                direction="positive" if coef > 0 else "negative",
                rank=0,
                interpretation=""
            ))

        contributions.sort(key=lambda x: abs(x.shapley_value), reverse=True)
        for i, contrib in enumerate(contributions):
            contrib.rank = i + 1

        return contributions

    # =========================================================================
    # 4. 가설 검증
    # =========================================================================

    def validate_hypotheses(
        self,
        hypotheses: List[Any],
        contributions: List[DriverContribution]
    ) -> List[HypothesisValidation]:
        """
        가설별 검증 수행

        검증 기준:
        - 기여도 상위 3위 이내 OR 기여도 >= 10% → validated
        - 그 외 → not_validated

        Args:
            hypotheses: Hypothesis 객체 리스트
            contributions: 계산된 기여도 리스트

        Returns:
            HypothesisValidation 리스트
        """
        # 기여도를 driver_id로 인덱싱
        contrib_map = {c.driver_id: c for c in contributions}

        validations = []

        for h in hypotheses:
            driver_id = getattr(h, 'driver_id', None) or getattr(h, 'driver', None) or getattr(h, 'factor', '')
            h_id = getattr(h, 'id', 'H?')
            description = getattr(h, 'description', '')
            expected_dir = getattr(h, 'direction', 'unknown')

            contrib = contrib_map.get(driver_id)

            if contrib is None:
                # Driver가 분석에 포함되지 않음
                validations.append(HypothesisValidation(
                    hypothesis_id=h_id,
                    driver_id=driver_id,
                    description=description,
                    expected_direction=expected_dir,
                    validation_status="not_validated",
                    confidence_score=0.0,
                    reasoning=f"Driver '{driver_id}'가 분석 데이터에 없습니다."
                ))
                continue

            # 검증 기준
            is_top3 = contrib.rank <= 3
            is_significant = contrib.contribution_pct >= 10.0

            if is_top3 or is_significant:
                status = "validated"
                confidence = min(0.5 + contrib.contribution_pct / 100, 1.0)
                reasoning = f"기여도 {contrib.contribution_pct:.1f}% (#{contrib.rank}위)"
            else:
                status = "not_validated"
                confidence = contrib.contribution_pct / 100
                reasoning = f"기여도 {contrib.contribution_pct:.1f}%로 유의미하지 않음 (#{contrib.rank}위)"

            validations.append(HypothesisValidation(
                hypothesis_id=h_id,
                driver_id=driver_id,
                description=description,
                expected_direction=expected_dir,
                validation_status=status,
                confidence_score=round(confidence, 2),
                reasoning=reasoning
            ))

        return validations

    # =========================================================================
    # 5. LLM 해석 생성 (Optional)
    # =========================================================================

    def generate_interpretation(
        self,
        kpi_change_pct: float,
        contributions: List[DriverContribution],
        validations: List[HypothesisValidation]
    ) -> str:
        """
        LLM으로 자연어 해석 생성 (Optional)

        llm_client가 없으면 간단한 템플릿 기반 해석 반환
        """
        if self.llm_client is None:
            return self._template_interpretation(kpi_change_pct, contributions, validations)

        # LLM 호출 (구현 필요 시 추가)
        return self._template_interpretation(kpi_change_pct, contributions, validations)

    def _template_interpretation(
        self,
        kpi_change_pct: float,
        contributions: List[DriverContribution],
        validations: List[HypothesisValidation]
    ) -> str:
        """템플릿 기반 해석 생성"""
        direction = "감소" if kpi_change_pct < 0 else "증가"

        top3 = contributions[:3]
        top3_text = ", ".join([
            f"{c.driver_name}({c.contribution_pct:.1f}%)" for c in top3
        ])

        validated_count = sum(1 for v in validations if v.validation_status == "validated")

        return (
            f"KPI가 {abs(kpi_change_pct):.1f}% {direction}했습니다. "
            f"주요 원인은 {top3_text}입니다. "
            f"총 {len(validations)}개 가설 중 {validated_count}개가 검증되었습니다."
        )

    # =========================================================================
    # 메인 진입점
    # =========================================================================

    def analyze(
        self,
        hypotheses: List[Any],
        kpi_id: str = "revenue",
        period: Dict = None,
        months: int = 24
    ) -> AnalysisResult:
        """
        전체 분석 실행

        1. 시계열 수집
        2. 모델 학습
        3. SHAP 계산
        4. 가설 검증
        5. 해석 생성

        Args:
            hypotheses: Hypothesis 객체 리스트
            kpi_id: 분석할 KPI ("revenue", "profit", "quantity")
            period: 분석 기간 {"year": 2024, "quarter": 4} (현재 미사용)
            months: 시계열 수집 개월 수

        Returns:
            AnalysisResult
        """
        # Driver ID 추출
        driver_ids = []
        for h in hypotheses:
            d_id = getattr(h, 'driver_id', None) or getattr(h, 'driver', None) or getattr(h, 'factor', None)
            if d_id and d_id in DRIVER_CONFIG:
                driver_ids.append(d_id)

        driver_ids = list(set(driver_ids))  # 중복 제거

        if len(driver_ids) < 2:
            return AnalysisResult(
                kpi_change_summary="분석 가능한 Driver가 부족합니다.",
                kpi_change_pct=0.0,
                top_drivers=[],
                hypotheses=[],
                final_explanation="최소 2개 이상의 Driver가 필요합니다.",
                model_r_squared=0.0,
                data_quality={"error": "insufficient_drivers", "found": len(driver_ids)}
            )

        # 1. 시계열 수집
        driver_series, kpi_series, quality_info = self.collect_all_series(
            driver_ids, kpi_id, months
        )

        if kpi_series is None or len(driver_series) < 2:
            return AnalysisResult(
                kpi_change_summary="데이터 수집에 실패했습니다.",
                kpi_change_pct=0.0,
                top_drivers=[],
                hypotheses=[],
                final_explanation="KPI 또는 Driver 시계열 데이터가 부족합니다.",
                model_r_squared=0.0,
                data_quality=quality_info
            )

        # KPI 변화율 계산
        if len(kpi_series) >= 2:
            kpi_change_pct = (kpi_series.iloc[-1] - kpi_series.iloc[-13]) / kpi_series.iloc[-13] * 100 \
                if len(kpi_series) >= 13 else \
                (kpi_series.iloc[-1] - kpi_series.iloc[0]) / kpi_series.iloc[0] * 100
        else:
            kpi_change_pct = 0.0

        # 2. 모델 학습
        try:
            model, scaler, driver_names, r_squared = self.build_model(driver_series, kpi_series)
        except Exception as e:
            return AnalysisResult(
                kpi_change_summary=f"모델 학습 실패: {str(e)}",
                kpi_change_pct=kpi_change_pct,
                top_drivers=[],
                hypotheses=[],
                final_explanation=str(e),
                model_r_squared=0.0,
                data_quality=quality_info
            )

        # 3. SHAP 기여도 계산
        contributions = self.calculate_contributions(
            model, scaler, driver_names, driver_series, kpi_series
        )

        # 4. 가설 검증
        validations = self.validate_hypotheses(hypotheses, contributions)

        # 5. 해석 생성
        direction = "감소" if kpi_change_pct < 0 else "증가"
        kpi_change_summary = f"KPI({kpi_id})가 전년 대비 {abs(kpi_change_pct):.1f}% {direction}했습니다."

        final_explanation = self.generate_interpretation(
            kpi_change_pct, contributions, validations
        )

        return AnalysisResult(
            kpi_change_summary=kpi_change_summary,
            kpi_change_pct=round(kpi_change_pct, 2),
            top_drivers=contributions,
            hypotheses=validations,
            final_explanation=final_explanation,
            model_r_squared=round(r_squared, 3),
            data_quality=quality_info
        )

    def to_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """AnalysisResult를 JSON-serializable dict로 변환"""
        return {
            "kpi_change_summary": result.kpi_change_summary,
            "kpi_change_pct": result.kpi_change_pct,
            "model_r_squared": result.model_r_squared,
            "data_quality": result.data_quality,
            "top_drivers": [
                {
                    "driver_id": d.driver_id,
                    "driver_name": d.driver_name,
                    "shapley_value": d.shapley_value,
                    "contribution_pct": d.contribution_pct,
                    "direction": d.direction,
                    "rank": d.rank,
                    "interpretation": d.interpretation
                }
                for d in result.top_drivers
            ],
            "hypotheses": [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "driver_id": h.driver_id,
                    "description": h.description,
                    "expected_direction": h.expected_direction,
                    "validation_status": h.validation_status,
                    "confidence_score": h.confidence_score,
                    "reasoning": h.reasoning
                }
                for h in result.hypotheses
            ],
            "final_explanation": result.final_explanation
        }

    def execute(self, hypotheses: List[Any], kpi_id: str = "revenue") -> ToolResult:
        """BaseTool 인터페이스 구현"""
        try:
            result = self.analyze(hypotheses, kpi_id)
            return ToolResult(
                success=True,
                data=self.to_dict(result)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"분석 실패: {str(e)}"
            )

    # =========================================================================
    # Stage 2: Plan 기반 실행 (3-Stage Architecture)
    # =========================================================================

    def execute_plan(
        self,
        plan: Dict[str, Any],
        hypotheses: List[Any] = None,
        kpi_id: str = "revenue"
    ) -> Dict[str, Any]:
        """
        Analysis Planner의 계획을 실행

        Args:
            plan: AnalysisPlanner.plan() 결과
            hypotheses: 가설 리스트 (선택)
            kpi_id: KPI 식별자

        Returns:
            Interpreter용 구조화된 결과 JSON
        """
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.linear_model import Ridge, Lasso

        # 1. Plan 파라미터 추출
        base_granularity = plan.get("base_granularity", "monthly")
        history_window = plan.get("history_window_months", 24)
        target_def = plan.get("target_definition", {})
        feature_plan = plan.get("feature_plan", {})
        modeling_plan = plan.get("modeling_plan", {})
        reporting_plan = plan.get("reporting_plan", {})

        data_warnings = {
            "high_correlation_groups": [],
            "merged_variables": [],
            "effective_months": 0
        }

        # 2. KPI 시계열 수집
        kpi_series = self.get_kpi_time_series(kpi_id, history_window)
        if kpi_series is None:
            return self._empty_result("KPI 데이터 수집 실패")

        data_warnings["effective_months"] = len(kpi_series)

        # 3. KPI 변환 적용
        target_type = target_def.get("type", "delta")
        kpi_transformed = self._apply_transformation(kpi_series, target_type)

        # 4. Driver 시계열 수집 및 변환
        selected_drivers = feature_plan.get("selected_drivers", [])
        merged_groups = feature_plan.get("merged_groups", [])

        driver_series = {}
        for driver_info in selected_drivers:
            driver_id = driver_info.get("driver_id")
            series = self.get_driver_time_series(driver_id, history_window)
            if series is not None:
                # Driver도 동일한 변환 적용
                driver_series[driver_id] = self._apply_transformation(series, target_type)

        # 병합 그룹 처리
        for group in merged_groups:
            new_id = group.get("new_feature_id")
            source_ids = group.get("source_driver_ids", [])

            # 소스 Driver들의 평균
            source_series = []
            for src_id in source_ids:
                s = self.get_driver_time_series(src_id, history_window)
                if s is not None:
                    source_series.append(self._apply_transformation(s, target_type))

            if source_series:
                merged = pd.concat(source_series, axis=1).mean(axis=1)
                merged.name = new_id
                driver_series[new_id] = merged

            data_warnings["merged_variables"].append({
                "new_id": new_id,
                "sources": source_ids
            })

        if len(driver_series) < 2:
            return self._empty_result("충분한 Driver 데이터가 없습니다")

        # 5. 모델 훈련
        model_type = modeling_plan.get("model_type", "ridge")
        use_shap = modeling_plan.get("use_shap", True)
        validation_type = modeling_plan.get("validation", {}).get("type", "time_series_cv")

        model, scaler, driver_names, metrics = self._train_model_with_cv(
            driver_series,
            kpi_transformed,
            model_type=model_type,
            validation_type=validation_type
        )

        # 6. 월별 SHAP 계산
        monthly_shap = []
        if use_shap and model is not None:
            monthly_shap = self._compute_monthly_shap(
                model, scaler, driver_names, driver_series, kpi_transformed, kpi_series
            )

        # 7. 분기별 집계
        quarterly_aggregated = []
        if reporting_plan.get("needs_quarterly_rollup", True):
            quarterly_aggregated = self._aggregate_to_quarterly(monthly_shap)

        # 8. 가설 정보 포함
        hypotheses_info = []
        if hypotheses:
            for h in hypotheses:
                h_id = getattr(h, 'id', 'H?')
                driver_id = getattr(h, 'driver_id', None) or getattr(h, 'factor', '')
                description = getattr(h, 'description', '')
                direction = getattr(h, 'direction', 'unknown')

                hypotheses_info.append({
                    "hypothesis_id": h_id,
                    "driver_id": driver_id,
                    "description": description,
                    "expected_direction": f"{direction}_{'negative' if 'decrease' in str(direction).lower() else 'positive'}"
                })

        return {
            "model_metrics": metrics,
            "data_warnings": data_warnings,
            "monthly_shap": monthly_shap,
            "quarterly_aggregated": quarterly_aggregated,
            "hypotheses": hypotheses_info
        }

    def _apply_transformation(
        self,
        series: pd.Series,
        transform_type: str
    ) -> pd.Series:
        """
        시계열 변환 적용

        Args:
            series: 원본 시계열
            transform_type: "level", "delta", "yoy", "pct_change", "log"

        Returns:
            변환된 시계열
        """
        if series is None or len(series) == 0:
            return series

        if transform_type == "level":
            return series

        elif transform_type == "delta":
            return series.diff().dropna()

        elif transform_type == "pct_change":
            return series.pct_change().dropna()

        elif transform_type == "yoy":
            # 전년 동월 대비
            if len(series) >= 13:
                return (series / series.shift(12) - 1).dropna()
            else:
                # 데이터 부족시 delta로 fallback
                return series.diff().dropna()

        elif transform_type == "log":
            # 로그 변환 (양수만)
            positive_series = series.clip(lower=0.01)
            return np.log(positive_series)

        else:
            return series

    def _train_model_with_cv(
        self,
        driver_series: Dict[str, pd.Series],
        kpi_series: pd.Series,
        model_type: str = "ridge",
        validation_type: str = "time_series_cv"
    ) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
        """
        교차검증 포함 모델 훈련

        Returns:
            (model, scaler, driver_names, metrics_dict)
        """
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        # 공통 기간 추출
        common_index = kpi_series.index
        for series in driver_series.values():
            common_index = common_index.intersection(series.index)

        if len(common_index) < 12:
            return None, None, [], {
                "num_samples": len(common_index),
                "num_features": len(driver_series),
                "train_r2": 0.0,
                "test_r2": 0.0,
                "cv_mean_r2": 0.0,
                "cv_std_r2": 0.0,
                "error": "샘플 수 부족"
            }

        # DataFrame 구성
        X = pd.DataFrame({
            name: series.loc[common_index].values
            for name, series in driver_series.items()
        })
        y = kpi_series.loc[common_index].values

        # 결측치 처리
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        driver_names = list(X.columns)

        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 모델 선택
        if model_type == "lasso":
            model = Lasso(alpha=1.0, max_iter=2000)
        else:
            model = Ridge(alpha=1.0)

        # 교차검증
        metrics = {
            "num_samples": len(common_index),
            "num_features": len(driver_names),
            "train_r2": 0.0,
            "test_r2": 0.0,
            "cv_mean_r2": 0.0,
            "cv_std_r2": 0.0
        }

        if validation_type == "time_series_cv" and len(common_index) >= 15:
            n_splits = min(3, len(common_index) // 5)
            if n_splits >= 2:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
                metrics["cv_mean_r2"] = round(float(np.mean(cv_scores)), 3)
                metrics["cv_std_r2"] = round(float(np.std(cv_scores)), 3)

        # 전체 데이터로 최종 훈련
        model.fit(X_scaled, y)
        metrics["train_r2"] = round(float(model.score(X_scaled, y)), 3)

        # Test set (마지막 20%)
        test_size = max(3, len(common_index) // 5)
        X_test = X_scaled[-test_size:]
        y_test = y[-test_size:]
        metrics["test_r2"] = round(float(model.score(X_test, y_test)), 3)

        return model, scaler, driver_names, metrics

    def _compute_monthly_shap(
        self,
        model,
        scaler,
        driver_names: List[str],
        driver_series: Dict[str, pd.Series],
        kpi_transformed: pd.Series,
        kpi_original: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        월별 SHAP 값 계산

        Returns:
            월별 SHAP 결과 리스트
        """
        try:
            import shap
        except ImportError:
            return self._fallback_monthly_contributions(
                model, scaler, driver_names, driver_series, kpi_transformed
            )

        # 공통 기간
        common_index = kpi_transformed.index
        for series in driver_series.values():
            common_index = common_index.intersection(series.index)

        if len(common_index) < 6:
            return []

        # DataFrame 구성
        X = pd.DataFrame({
            name: driver_series[name].loc[common_index].values
            for name in driver_names
        })
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 스케일링
        X_scaled = scaler.transform(X)

        # SHAP Explainer
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

        # 월별 결과 구성
        monthly_results = []

        for i, month in enumerate(common_index):
            # KPI 변화 계산
            kpi_change = float(kpi_transformed.loc[month]) if month in kpi_transformed.index else 0

            # 해당 월의 SHAP 값
            month_shap = shap_values[i]
            total_abs = sum(abs(v) for v in month_shap)

            drivers_contrib = []
            for j, driver_id in enumerate(driver_names):
                shap_val = float(month_shap[j])
                contrib_pct = abs(shap_val) / total_abs * 100 if total_abs > 0 else 0

                drivers_contrib.append({
                    "driver_id": driver_id,
                    "shap": round(shap_val, 6),
                    "contribution_pct": round(contrib_pct, 1),
                    "rank": 0  # 나중에 설정
                })

            # 순위 부여
            drivers_contrib.sort(key=lambda x: abs(x["shap"]), reverse=True)
            for rank, d in enumerate(drivers_contrib):
                d["rank"] = rank + 1

            monthly_results.append({
                "month": str(month),
                "kpi_change": round(kpi_change, 4),
                "drivers": drivers_contrib
            })

        return monthly_results

    def _fallback_monthly_contributions(
        self,
        model,
        scaler,
        driver_names: List[str],
        driver_series: Dict[str, pd.Series],
        kpi_transformed: pd.Series
    ) -> List[Dict[str, Any]]:
        """SHAP 없이 회귀계수 기반 월별 기여도 계산 (fallback)"""
        coefficients = model.coef_
        total = sum(abs(c) for c in coefficients)

        # 공통 기간
        common_index = kpi_transformed.index
        for series in driver_series.values():
            common_index = common_index.intersection(series.index)

        monthly_results = []
        for month in common_index:
            kpi_change = float(kpi_transformed.loc[month]) if month in kpi_transformed.index else 0

            drivers_contrib = []
            for i, driver_id in enumerate(driver_names):
                coef = float(coefficients[i])
                contrib_pct = abs(coef) / total * 100 if total > 0 else 0

                drivers_contrib.append({
                    "driver_id": driver_id,
                    "shap": round(coef, 6),
                    "contribution_pct": round(contrib_pct, 1),
                    "rank": i + 1
                })

            # 순위 부여
            drivers_contrib.sort(key=lambda x: abs(x["shap"]), reverse=True)
            for rank, d in enumerate(drivers_contrib):
                d["rank"] = rank + 1

            monthly_results.append({
                "month": str(month),
                "kpi_change": round(kpi_change, 4),
                "drivers": drivers_contrib
            })

        return monthly_results

    def _aggregate_to_quarterly(
        self,
        monthly_shap: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        월별 SHAP 결과를 분기별로 집계

        Args:
            monthly_shap: _compute_monthly_shap() 결과

        Returns:
            분기별 집계 결과 리스트
        """
        if not monthly_shap:
            return []

        # 월별 데이터를 분기로 그룹화
        quarterly_data = {}

        for month_data in monthly_shap:
            month_str = month_data.get("month", "")
            if not month_str or len(month_str) < 7:
                continue

            # 분기 계산
            try:
                year = month_str[:4]
                month_num = int(month_str[5:7])
                quarter = (month_num - 1) // 3 + 1
                quarter_key = f"{year}-Q{quarter}"
            except:
                continue

            if quarter_key not in quarterly_data:
                quarterly_data[quarter_key] = {
                    "months": [],
                    "kpi_changes": [],
                    "driver_contributions": {}
                }

            quarterly_data[quarter_key]["months"].append(month_str)
            quarterly_data[quarter_key]["kpi_changes"].append(month_data.get("kpi_change", 0))

            # Driver별 기여도 수집
            for driver in month_data.get("drivers", []):
                driver_id = driver.get("driver_id")
                if driver_id not in quarterly_data[quarter_key]["driver_contributions"]:
                    quarterly_data[quarter_key]["driver_contributions"][driver_id] = {
                        "shap_values": [],
                        "contribution_pcts": [],
                        "ranks": []
                    }

                quarterly_data[quarter_key]["driver_contributions"][driver_id]["shap_values"].append(driver.get("shap", 0))
                quarterly_data[quarter_key]["driver_contributions"][driver_id]["contribution_pcts"].append(driver.get("contribution_pct", 0))
                quarterly_data[quarter_key]["driver_contributions"][driver_id]["ranks"].append(driver.get("rank", 999))

        # 분기별 결과 생성
        quarterly_results = []

        for quarter_key in sorted(quarterly_data.keys()):
            qdata = quarterly_data[quarter_key]

            # KPI 총 변화
            kpi_change_total = sum(qdata["kpi_changes"])

            # Driver별 집계
            drivers_summary = []
            for driver_id, contrib_data in qdata["driver_contributions"].items():
                avg_contribution = np.mean(contrib_data["contribution_pcts"])
                months_in_top3 = sum(1 for r in contrib_data["ranks"] if r <= 3)

                drivers_summary.append({
                    "driver_id": driver_id,
                    "avg_contribution_pct": round(avg_contribution, 1),
                    "months_in_top3": months_in_top3
                })

            # 평균 기여도 기준 정렬
            drivers_summary.sort(key=lambda x: x["avg_contribution_pct"], reverse=True)

            quarterly_results.append({
                "quarter": quarter_key,
                "kpi_change_total": round(kpi_change_total, 4),
                "drivers": drivers_summary[:10]  # 상위 10개만
            })

        return quarterly_results

    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """빈 결과 반환"""
        return {
            "model_metrics": {
                "num_samples": 0,
                "num_features": 0,
                "train_r2": 0.0,
                "test_r2": 0.0,
                "cv_mean_r2": 0.0,
                "cv_std_r2": 0.0,
                "error": error_message
            },
            "data_warnings": {
                "high_correlation_groups": [],
                "merged_variables": [],
                "effective_months": 0
            },
            "monthly_shap": [],
            "quarterly_aggregated": [],
            "hypotheses": []
        }
