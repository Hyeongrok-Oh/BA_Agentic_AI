"""
Metadata Collector Tool - Analysis Planner를 위한 메타데이터 수집

주요 기능:
1. KPI/Driver 시계열 수집
2. 상관계수 계산 (correlation_with_kpi)
3. VIF 계산 (다중공선성)
4. 데이터 품질 평가 (missing_pct, outliers)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from ..base import BaseTool, ToolResult


# =============================================================================
# Driver-Table 매핑 (data_analysis.py와 동일)
# =============================================================================

DRIVER_CONFIG = {
    # T1: ERP 직접 검증
    "출하량": {"table": "TR_SALES", "column": "QTY", "date_col": "SALES_DATE", "category": "sales"},
    "판매량": {"table": "TR_SALES", "column": "QTY", "date_col": "SALES_DATE", "category": "sales"},
    "OLED비중": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE", "filter": "prod.DISPLAY_TYPE = 'OLED'", "category": "product_mix"},
    "프리미엄비중": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE", "filter": "prod.IS_PREMIUM = 'Y'", "category": "product_mix"},
    "TV평균판매가": {"table": "TR_SALES", "column": "REVENUE_USD/QTY", "date_col": "SALES_DATE", "category": "pricing"},
    "할인율": {"table": "TR_EXPENSE", "column": "PROMOTION_COST", "date_col": "EXPENSE_DATE", "category": "pricing"},
    "프로모션비용": {"table": "TR_EXPENSE", "column": "PROMOTION_COST", "date_col": "EXPENSE_DATE", "category": "cost"},
    "패널원가": {"table": "TR_PURCHASE", "column": "PANEL_PRICE_USD", "date_col": "PURCHASE_DATE", "category": "cost"},
    "제조원가": {"table": "TR_PURCHASE", "column": "TOTAL_COGS_USD", "date_col": "PURCHASE_DATE", "category": "cost"},
    "물류비": {"table": "TR_EXPENSE", "column": "LOGISTICS_COST", "date_col": "EXPENSE_DATE", "category": "cost"},
    "원재료비": {"table": "TR_PURCHASE", "column": "RAW_MATERIAL_INDEX", "date_col": "PURCHASE_DATE", "category": "cost"},
    "달러환율": {"table": "EXT_MACRO", "column": "EXCHANGE_RATE_KRW_USD", "date_col": "DATA_DATE", "category": "macro"},
    # T2: Proxy
    "글로벌TV수요": {"table": "EXT_MARKET", "column": "TOTAL_SHIPMENT_10K", "date_col": "DATA_DATE", "category": "market"},
    "소비심리": {"table": "EXT_MACRO", "column": "CSI_INDEX", "date_col": "DATA_DATE", "category": "macro"},
    "인플레이션": {"table": "EXT_MACRO", "column": "INFLATION_RATE", "date_col": "DATA_DATE", "category": "macro"},
    "금리": {"table": "EXT_MACRO", "column": "INTEREST_RATE", "date_col": "DATA_DATE", "category": "macro"},
}

KPI_CONFIG = {
    "revenue": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE", "name_kr": "매출"},
    "profit": {"table": "TR_SALES", "column": "REVENUE_USD", "date_col": "SALES_DATE", "name_kr": "영업이익"},
    "quantity": {"table": "TR_SALES", "column": "QTY", "date_col": "SALES_DATE", "name_kr": "판매량"},
}

# VIF 및 상관계수 임계값
VIF_THRESHOLD_WARNING = 5.0
VIF_THRESHOLD_HIGH = 10.0
CORRELATION_HIGH_THRESHOLD = 0.7


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DriverMetadata:
    """Driver 메타데이터"""
    driver_id: str
    name: str
    category: str
    data_quality: Dict[str, float]  # missing_pct, outlier_count, variance
    correlation_with_kpi: float
    correlation_with_kpi_delta: float
    vif: float
    high_corr_group: Optional[str] = None


@dataclass
class KPIMetadata:
    """KPI 메타데이터"""
    id: str
    name_kr: str
    time_range: Dict[str, str]  # start, end
    granularity_available: List[str]
    trend_strength: str  # low, medium, high
    seasonality_strength: str  # low, medium, high


@dataclass
class CollectedMetadata:
    """수집된 전체 메타데이터"""
    kpi: KPIMetadata
    drivers: List[DriverMetadata]
    data_quality_summary: Dict[str, Any]
    high_correlation_groups: List[List[str]]


# =============================================================================
# Main Class
# =============================================================================

class MetadataCollector(BaseTool):
    """
    Analysis Planner를 위한 메타데이터 수집기

    사용법:
        collector = MetadataCollector(sql_executor)
        metadata = collector.collect(kpi_id="revenue", driver_ids=[...], months=36)
    """

    name = "metadata_collector"
    description = "KPI/Driver 메타데이터 및 통계 정보 수집"

    def __init__(self, sql_executor):
        """
        Args:
            sql_executor: SQLExecutor 인스턴스
        """
        self.sql_executor = sql_executor
        self._series_cache: Dict[str, pd.Series] = {}

    # =========================================================================
    # 1. 시계열 데이터 수집
    # =========================================================================

    def _get_driver_series(
        self,
        driver_id: str,
        months: int = 36
    ) -> Optional[pd.Series]:
        """Driver 월별 시계열 추출"""
        cache_key = f"driver_{driver_id}_{months}"
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        config = DRIVER_CONFIG.get(driver_id)
        if not config:
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
            return None

        series = pd.Series(
            result.data['value'].values,
            index=pd.Index(result.data['period'].values, name='period'),
            name=driver_id
        )

        self._series_cache[cache_key] = series
        return series

    def _get_kpi_series(
        self,
        kpi_id: str = "revenue",
        months: int = 36
    ) -> Optional[pd.Series]:
        """KPI 월별 시계열 추출"""
        cache_key = f"kpi_{kpi_id}_{months}"
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        config = KPI_CONFIG.get(kpi_id)
        if not config:
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

    # =========================================================================
    # 2. 통계 계산
    # =========================================================================

    def _calculate_correlations(
        self,
        driver_series: Dict[str, pd.Series],
        kpi_series: pd.Series
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Driver-KPI 상관계수 계산

        Returns:
            (level_correlations, delta_correlations)
        """
        level_corr = {}
        delta_corr = {}

        # 공통 기간 찾기
        common_index = kpi_series.index
        for series in driver_series.values():
            common_index = common_index.intersection(series.index)

        if len(common_index) < 12:
            return {}, {}

        kpi_aligned = kpi_series.loc[common_index]
        kpi_delta = kpi_aligned.diff().dropna()

        for driver_id, series in driver_series.items():
            driver_aligned = series.loc[common_index]
            driver_delta = driver_aligned.diff().dropna()

            # Level correlation
            try:
                level_corr[driver_id] = float(driver_aligned.corr(kpi_aligned))
            except:
                level_corr[driver_id] = 0.0

            # Delta correlation
            try:
                # Delta 인덱스 맞추기
                common_delta_idx = kpi_delta.index.intersection(driver_delta.index)
                if len(common_delta_idx) >= 10:
                    delta_corr[driver_id] = float(
                        driver_delta.loc[common_delta_idx].corr(kpi_delta.loc[common_delta_idx])
                    )
                else:
                    delta_corr[driver_id] = 0.0
            except:
                delta_corr[driver_id] = 0.0

        return level_corr, delta_corr

    def _calculate_vif(
        self,
        driver_series: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        VIF (Variance Inflation Factor) 계산

        VIF > 5: 다중공선성 주의
        VIF > 10: 심각, 변수 제거/병합 고려
        """
        if len(driver_series) < 2:
            return {d: 1.0 for d in driver_series}

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            # statsmodels 없으면 기본값 반환
            return {d: 1.0 for d in driver_series}

        # 공통 기간 찾기
        common_index = None
        for series in driver_series.values():
            if common_index is None:
                common_index = series.index
            else:
                common_index = common_index.intersection(series.index)

        if common_index is None or len(common_index) < 10:
            return {d: 1.0 for d in driver_series}

        # DataFrame 구성
        df = pd.DataFrame({
            name: series.loc[common_index].values
            for name, series in driver_series.items()
        })

        # 결측치 처리
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 표준화
        df_std = (df - df.mean()) / df.std().replace(0, 1)

        vifs = {}
        driver_names = list(driver_series.keys())

        for i, col in enumerate(driver_names):
            try:
                vif_val = variance_inflation_factor(df_std.values, i)
                vifs[col] = round(float(vif_val), 2) if not np.isinf(vif_val) else 100.0
            except:
                vifs[col] = 1.0

        return vifs

    def _find_high_correlation_groups(
        self,
        driver_series: Dict[str, pd.Series],
        threshold: float = CORRELATION_HIGH_THRESHOLD
    ) -> Tuple[List[List[str]], Dict[str, str]]:
        """
        높은 상관관계를 가진 Driver 그룹 찾기

        Returns:
            (correlation_groups, driver_to_group_map)
        """
        if len(driver_series) < 2:
            return [], {}

        # 공통 기간
        common_index = None
        for series in driver_series.values():
            if common_index is None:
                common_index = series.index
            else:
                common_index = common_index.intersection(series.index)

        if common_index is None or len(common_index) < 10:
            return [], {}

        # 상관행렬 계산
        df = pd.DataFrame({
            name: series.loc[common_index].values
            for name, series in driver_series.items()
        })

        corr_matrix = df.corr()

        # Union-Find로 그룹화
        parent = {d: d for d in driver_series}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        drivers = list(driver_series.keys())
        for i, d1 in enumerate(drivers):
            for j, d2 in enumerate(drivers):
                if i < j:
                    corr_val = abs(corr_matrix.loc[d1, d2])
                    if corr_val >= threshold:
                        union(d1, d2)

        # 그룹 생성
        groups_dict = {}
        for d in drivers:
            root = find(d)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(d)

        # 2개 이상인 그룹만 반환
        groups = [sorted(g) for g in groups_dict.values() if len(g) > 1]

        # Driver → Group 매핑
        driver_to_group = {}
        for i, group in enumerate(groups):
            group_name = f"G{i+1}"
            for d in group:
                driver_to_group[d] = group_name

        return groups, driver_to_group

    # =========================================================================
    # 3. 데이터 품질 평가
    # =========================================================================

    def _assess_data_quality(
        self,
        series: pd.Series
    ) -> Dict[str, float]:
        """개별 시계열 데이터 품질 평가"""
        if series is None or len(series) == 0:
            return {"missing_pct": 100.0, "outlier_count": 0, "variance": 0.0}

        # 결측치 비율
        missing_pct = series.isna().sum() / len(series) * 100

        # 이상치 개수 (IQR 방식)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())

        # 변동성 (표준편차 / 평균)
        mean_val = series.mean()
        std_val = series.std()
        variance = std_val / abs(mean_val) if mean_val != 0 else 0.0

        return {
            "missing_pct": round(missing_pct, 1),
            "outlier_count": outlier_count,
            "variance": round(variance, 3)
        }

    def _assess_trend_seasonality(
        self,
        series: pd.Series
    ) -> Tuple[str, str]:
        """
        추세 및 계절성 강도 평가

        Returns:
            (trend_strength, seasonality_strength)
        """
        if series is None or len(series) < 12:
            return "low", "low"

        # 추세 강도: 선형 회귀 R²
        try:
            x = np.arange(len(series))
            y = series.values

            # 결측치 제거
            mask = ~np.isnan(y)
            x, y = x[mask], y[mask]

            if len(x) < 6:
                return "low", "low"

            # 선형 회귀
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r_squared > 0.6:
                trend = "high"
            elif r_squared > 0.3:
                trend = "medium"
            else:
                trend = "low"
        except:
            trend = "low"

        # 계절성 강도: 12개월 자기상관
        try:
            if len(series) >= 24:
                lag12_corr = series.autocorr(lag=12)
                if lag12_corr > 0.5:
                    seasonality = "high"
                elif lag12_corr > 0.25:
                    seasonality = "medium"
                else:
                    seasonality = "low"
            else:
                seasonality = "low"
        except:
            seasonality = "low"

        return trend, seasonality

    # =========================================================================
    # 메인 수집 메서드
    # =========================================================================

    def collect(
        self,
        kpi_id: str,
        driver_ids: List[str],
        months: int = 36
    ) -> Dict[str, Any]:
        """
        Analysis Planner를 위한 전체 메타데이터 수집

        Args:
            kpi_id: KPI 식별자 ("revenue", "profit", "quantity")
            driver_ids: Driver 식별자 리스트 (한글)
            months: 수집할 개월 수

        Returns:
            Planner 입력용 메타데이터 JSON
        """
        # 1. KPI 시계열 수집
        kpi_series = self._get_kpi_series(kpi_id, months)

        if kpi_series is None:
            return {
                "error": f"KPI '{kpi_id}' 데이터 수집 실패",
                "kpi": None,
                "drivers": [],
                "data_quality_summary": {}
            }

        # 2. Driver 시계열 수집
        driver_series = {}
        for driver_id in driver_ids:
            series = self._get_driver_series(driver_id, months)
            if series is not None and len(series) >= 12:
                driver_series[driver_id] = series

        if len(driver_series) < 2:
            return {
                "error": "충분한 Driver 데이터가 없습니다",
                "kpi": None,
                "drivers": [],
                "data_quality_summary": {"valid_drivers": len(driver_series)}
            }

        # 3. 상관계수 계산
        level_corr, delta_corr = self._calculate_correlations(driver_series, kpi_series)

        # 4. VIF 계산
        vifs = self._calculate_vif(driver_series)

        # 5. 고상관 그룹 찾기
        high_corr_groups, driver_to_group = self._find_high_correlation_groups(driver_series)

        # 6. KPI 추세/계절성 평가
        trend_strength, seasonality_strength = self._assess_trend_seasonality(kpi_series)

        # 7. Driver 메타데이터 구성
        drivers_metadata = []
        drivers_with_issues = []

        for driver_id, series in driver_series.items():
            config = DRIVER_CONFIG.get(driver_id, {})
            quality = self._assess_data_quality(series)

            # 이슈 있는 driver 체크
            if quality["missing_pct"] > 10 or vifs.get(driver_id, 1) > VIF_THRESHOLD_HIGH:
                drivers_with_issues.append(driver_id)

            drivers_metadata.append({
                "driver_id": driver_id,
                "name": driver_id,
                "category": config.get("category", "unknown"),
                "missing_rate": quality["missing_pct"] / 100,
                "corr_with_kpi_level": round(level_corr.get(driver_id, 0), 3),
                "corr_with_kpi_delta": round(delta_corr.get(driver_id, 0), 3),
                "vif": vifs.get(driver_id, 1.0),
                "high_corr_group": driver_to_group.get(driver_id)
            })

        # 8. 최종 메타데이터 구성
        return {
            "kpi": {
                "id": kpi_id,
                "name": KPI_CONFIG.get(kpi_id, {}).get("name_kr", kpi_id),
                "type": "flow",  # 매출, 판매량 등은 flow
                "preferred_reporting": "quarterly"
            },
            "time_info": {
                "available_granularities": ["monthly", "quarterly"],
                "monthly": {
                    "num_samples": len(kpi_series),
                    "start": str(kpi_series.index[0]) if len(kpi_series) > 0 else None,
                    "end": str(kpi_series.index[-1]) if len(kpi_series) > 0 else None
                },
                "quarterly": {
                    "num_samples": len(kpi_series) // 3,
                    "start": None,
                    "end": None
                }
            },
            "drivers": drivers_metadata,
            "data_quality": {
                "trend_strength": trend_strength,
                "seasonality_strength": seasonality_strength,
                "outlier_months": [],  # 추후 구현
                "effective_months": len(kpi_series)
            },
            "business_constraints": {
                "need_quarterly_view": True,
                "min_history_months": 24
            },
            "high_correlation_groups": high_corr_groups,
            "data_quality_summary": {
                "total_drivers": len(driver_ids),
                "valid_drivers": len(driver_series),
                "drivers_with_issues": drivers_with_issues
            }
        }

    def execute(self, kpi_id: str, driver_ids: List[str], months: int = 36) -> ToolResult:
        """BaseTool 인터페이스 구현"""
        try:
            result = self.collect(kpi_id, driver_ids, months)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=f"메타데이터 수집 실패: {str(e)}")
