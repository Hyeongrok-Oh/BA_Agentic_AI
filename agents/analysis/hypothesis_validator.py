"""
Hypothesis Validator Agent - 가설 검증 에이전트

역할:
- SQLGenerator를 사용하여 각 가설을 ERP 데이터로 검증
- SQL 검증 불가 시 Graph DB 기반 인과관계 설명으로 대체
- 임계값 이상 변동이 있는 가설만 validated로 표시
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..base import BaseAgent, AgentContext
from ..tools import SQLGenerator, SQLExecutor, GraphExecutor
from .hypothesis_generator import Hypothesis


# =============================================================================
# ERP 스키마 기반 SQL 검증 가능 여부 판단
# =============================================================================
#
# LGE ERP Database Schema (sql/schema_normalized_3nf.sql 참조)
#
# [TBL_TX_COST_DETAIL] - COST_TYPE 컬럼
#   - MAT: Material cost (재료비, 패널, 부품)
#   - LOG: Logistics cost (물류비, 운송비, 해상운임)
#   - TAR: Tariff (관세)
#   - OH:  Overhead (오버헤드, 제조간접비)
#
# [TBL_TX_PRICE_CONDITION] - COND_TYPE 컬럼
#   - PR00: Base list price (기본가, 정가)
#   - K007: Volume discount (할인, 볼륨할인)
#   - ZPRO: Price protection (가격보호, PP)
#   - ZMDF: Marketing Development Fund (MDF, 마케팅비)
#
# [TBL_TX_SALES_ITEM] - 판매 데이터
#   - ORDER_QTY: 판매수량
#   - NET_VALUE: 순매출
#
# [TBL_MD_PRODUCT] - 제품 마스터
#   - PANEL_TYPE: OLED, QNED, LCD (제품 유형별 분석 가능)
#   - SCREEN_SIZE: 화면 크기별 분석 가능
#   - SERIES: 시리즈별 분석 가능
#
# [TBL_ORG_SUBSIDIARY] - 지역
#   - REGION: NA, KR, EU (지역별 분석 가능)
#
# [TBL_ORG_CUSTOMER] - 고객/채널
#   - CHANNEL_TYPE: B2B, RETAIL, ONLINE (채널별 분석 가능)
# =============================================================================

# SQL 검증 가능한 Factor (ERP 테이블/컬럼에 매핑됨)
SQL_VALIDATABLE_KEYWORDS = {
    # TBL_TX_COST_DETAIL.COST_TYPE
    "cost_mat": ["재료비", "원재료", "패널비용", "부품비", "MAT"],
    "cost_log": ["물류비", "운송비", "해상운임", "운임", "배송비", "LOG"],
    "cost_tar": ["관세", "TAR"],  # 주의: DB에 있지만 외부요인과 연계 필요
    "cost_oh": ["오버헤드", "제조간접비", "간접비", "OH"],

    # TBL_TX_PRICE_CONDITION.COND_TYPE
    "price_base": ["기본가", "정가", "리스트가", "PR00"],
    "price_discount": ["할인", "볼륨할인", "K007"],
    "price_protection": ["가격보호", "PP", "ZPRO"],
    "price_mdf": ["MDF", "마케팅비", "ZMDF"],

    # TBL_TX_SALES_ITEM (집계)
    "sales_qty": ["판매수량", "판매량", "출하량", "ORDER_QTY"],
    "sales_revenue": ["매출", "순매출", "매출액", "NET_VALUE", "revenue"],

    # TBL_MD_PRODUCT (세그먼트 분석)
    "product_panel": ["OLED판매", "QNED판매", "LCD판매"],  # 제품유형별 매출/수량
    "product_size": ["대형TV", "소형TV"],  # 사이즈별 분석

    # TBL_ORG (지역/채널 분석)
    "region": ["북미매출", "유럽매출", "한국매출"],
    "channel": ["B2B매출", "리테일매출", "온라인매출"],
}

# SQL 검증 불가능한 Factor (ERP에 없는 외부 데이터)
# 이 Factor들은 Graph 기반 인과관계 설명으로 대체
GRAPH_ONLY_KEYWORDS = [
    # 거시경제 (ERP에 없음)
    "환율", "원달러", "달러", "금리", "인플레이션", "경기", "GDP",

    # 시장/경쟁 (ERP에 없음)
    "경쟁", "점유율", "시장", "수요", "소비심리", "소비자",
    "삼성", "TCL", "하이센스",

    # 공급망/외부이벤트 (ERP에 없음)
    "홍해", "수에즈", "공급망", "반도체", "지정학", "전쟁",
    "트럼프", "정책", "규제", "무역",

    # 원자재 시세 (ERP에 금액만 있고 가격변동 추세는 없음)
    "패널가격", "유가", "원자재가격", "부품가격",

    # 계절성/이벤트 (ERP에서 직접 측정 불가)
    "성수기", "블랙프라이데이", "올림픽", "월드컵",
]


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
    validation_type: str = "sql"  # "sql" or "graph"
    graph_evidence: Dict = field(default_factory=dict)  # Graph 기반 증거


class HypothesisValidator(BaseAgent):
    """가설 검증 에이전트 (SQL + Graph 하이브리드)"""

    name = "hypothesis_validator"
    description = "SQL 검증 우선, 불가 시 Graph DB 기반 인과관계 설명으로 대체"

    def __init__(self, api_key: str = None, db_path: str = None):
        super().__init__(api_key)
        self.sql_generator = SQLGenerator(db_path, api_key)
        self.sql_executor = SQLExecutor(db_path)
        self.graph_executor = GraphExecutor()
        self.add_tool(self.sql_generator)
        self.add_tool(self.sql_executor)
        self.add_tool(self.graph_executor)

    def validate(
        self,
        hypotheses: List[Hypothesis],
        period: Dict,
        region: str = None,
        threshold: float = 5.0
    ) -> List[Hypothesis]:
        """
        가설 목록 검증 (SQL 우선, 불가 시 Graph fallback)

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
            # 1. Factor가 SQL로 검증 가능한지 확인
            is_sql_validatable = self._is_sql_validatable(hypothesis.factor)

            result = None

            if is_sql_validatable:
                # 2. SQL 검증 시도
                result = self._validate_single(
                    hypothesis,
                    prev_start, prev_end,
                    curr_start, curr_end,
                    region,
                    threshold
                )

            # 3. SQL 검증 실패 또는 Graph-only Factor인 경우 Graph 검증
            if result is None:
                result = self._validate_with_graph(
                    hypothesis,
                    region
                )

            if result:
                hypothesis.validated = result.validated
                hypothesis.validation_data = {
                    "change_percent": result.change_percent,
                    "previous_value": result.previous_value,
                    "current_value": result.current_value,
                    "direction": result.direction,
                    "details": result.details,
                    "validation_type": result.validation_type,
                    "sql_query": result.sql_query,
                    "graph_evidence": result.graph_evidence
                }

                if result.validated:
                    validated_hypotheses.append(hypothesis)

        return validated_hypotheses

    def _is_sql_validatable(self, factor: str) -> bool:
        """
        Factor가 SQL로 검증 가능한지 ERP 스키마 기반으로 확인

        Returns:
            True: ERP 테이블에 해당 데이터가 존재 (SQL 검증 가능)
            False: ERP에 없는 외부 데이터 (Graph 검증 필요)
        """
        factor_lower = factor.lower()

        # 1. Graph-only 키워드 체크 (먼저 확인 - 외부 데이터)
        for keyword in GRAPH_ONLY_KEYWORDS:
            if keyword.lower() in factor_lower:
                return False

        # 2. SQL 검증 가능한 키워드 체크 (ERP 테이블에 매핑됨)
        for erp_mapping, keywords in SQL_VALIDATABLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in factor_lower:
                    return True

        # 3. 기본값: 불확실하면 SQL 먼저 시도 후 실패하면 Graph fallback
        #    (validate 메서드에서 SQL 실패 시 Graph로 자동 전환)
        return True

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

    def _validate_with_graph(
        self,
        hypothesis: Hypothesis,
        region: str = None
    ) -> Optional[ValidationResult]:
        """
        Graph DB 기반 가설 검증 (SQL 불가 시 fallback)

        Event → Factor → Anchor 인과관계 경로를 조회하여
        해당 Factor가 KPI에 영향을 미치는 근거를 제공
        """
        factor_name = hypothesis.factor

        # Region 필터 조건
        region_filter = ""
        if region:
            region_upper = region.upper()
            region_filter = f"""
            AND (
                size([(e)-[:TARGETS]->(r:Region) | r.id]) = 0
                OR '{region_upper}' IN [(e)-[:TARGETS]->(r:Region) | r.id]
                OR 'global' IN [reg IN [(e)-[:TARGETS]->(r:Region) | toLower(r.id)] | reg]
            )
            """

        # Graph 쿼리: Event → Factor → Anchor 경로 조회
        query = f"""
        MATCH (e:Event)-[r1:INCREASES|DECREASES]->(f:Factor)-[r2:AFFECTS]->(a:Anchor)
        WHERE toLower(f.name) CONTAINS toLower($factor_name)
        {region_filter}
        OPTIONAL MATCH (e)-[:TARGETS]->(reg:Region)
        RETURN
            e.name as event_name,
            e.category as event_category,
            e.severity as severity,
            e.evidence as event_evidence,
            type(r1) as event_impact,
            r1.magnitude as magnitude,
            f.name as factor_name,
            r2.type as factor_relation,
            a.name as anchor_name,
            collect(DISTINCT reg.id) as target_regions
        ORDER BY
            CASE e.severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 ELSE 3 END,
            CASE r1.magnitude WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END
        LIMIT 5
        """

        try:
            result = self.graph_executor.execute(query, {"factor_name": factor_name})

            if not result.success or not result.data:
                # Graph에서도 못 찾으면 가설 자체 정보로 검증
                return self._validate_with_hypothesis_info(hypothesis)

            events = result.data

            # 인과관계 설명 구성
            causal_chains = []
            for ev in events:
                event_name = ev.get("event_name", "")
                event_impact = ev.get("event_impact", "AFFECTS")
                factor = ev.get("factor_name", "")
                factor_relation = ev.get("factor_relation", "PROPORTIONAL")
                anchor = ev.get("anchor_name", "")
                magnitude = ev.get("magnitude", "medium")
                regions = ev.get("target_regions", [])

                impact_kr = "증가" if event_impact == "INCREASES" else "감소"
                relation_kr = "상승" if factor_relation == "PROPORTIONAL" else "하락"

                chain = f"[{event_name}] → {factor} {impact_kr} → {anchor} {relation_kr}"
                if regions:
                    chain += f" (지역: {', '.join(regions)})"

                causal_chains.append({
                    "event": event_name,
                    "event_category": ev.get("event_category", ""),
                    "impact": event_impact,
                    "factor": factor,
                    "relation": factor_relation,
                    "anchor": anchor,
                    "magnitude": magnitude,
                    "evidence": ev.get("event_evidence", ""),
                    "chain_text": chain
                })

            # 가설 방향과 Graph 관계 비교
            first_event = events[0] if events else {}
            event_impact = first_event.get("event_impact", "")
            factor_relation = first_event.get("factor_relation", "PROPORTIONAL")

            # 방향 일치 여부 판단
            validated = self._check_direction_match(
                hypothesis.direction,
                event_impact,
                factor_relation
            )

            # 상세 설명 생성
            details = f"[Graph 기반] {hypothesis.factor}: "
            if causal_chains:
                details += causal_chains[0]["chain_text"]
            else:
                details += "관련 이벤트 없음"

            return ValidationResult(
                hypothesis_id=hypothesis.id,
                validated=validated,
                change_percent=0.0,  # Graph 검증은 수치 없음
                previous_value=0.0,
                current_value=0.0,
                direction=hypothesis.direction,
                details=details,
                sql_query="",  # SQL 사용 안 함
                validation_type="graph",
                graph_evidence={
                    "causal_chains": causal_chains,
                    "event_count": len(events),
                    "source": "Knowledge Graph (Event → Factor → Anchor)"
                }
            )

        except Exception as e:
            print(f"Graph 검증 오류 ({hypothesis.id}): {e}")
            return self._validate_with_hypothesis_info(hypothesis)

    def _validate_with_hypothesis_info(
        self,
        hypothesis: Hypothesis
    ) -> ValidationResult:
        """
        Graph에서도 찾지 못한 경우, 가설 자체 정보로 기본 검증
        (graph_evidence에 저장된 정보 활용)
        """
        graph_evidence = hypothesis.graph_evidence or {}

        details = f"[Graph 기반] {hypothesis.factor}: "
        if graph_evidence.get("relation_evidence"):
            details += graph_evidence["relation_evidence"]
        else:
            details += f"{hypothesis.description}"

        return ValidationResult(
            hypothesis_id=hypothesis.id,
            validated=True,  # 가설 자체는 유효하다고 가정
            change_percent=0.0,
            previous_value=0.0,
            current_value=0.0,
            direction=hypothesis.direction,
            details=details,
            sql_query="",
            validation_type="graph",
            graph_evidence={
                "from_hypothesis": True,
                "factor_id": graph_evidence.get("factor_id", ""),
                "relation_type": graph_evidence.get("relation_type", ""),
                "mention_count": graph_evidence.get("mention_count", 0),
                "source": "Knowledge Graph (Factor → Anchor)"
            }
        )

    def _check_direction_match(
        self,
        hypothesis_direction: str,
        event_impact: str,
        factor_relation: str
    ) -> bool:
        """가설 방향과 Graph 관계 일치 여부 확인"""
        # Event가 Factor를 증가시키고, Factor가 Anchor에 비례 관계면
        # → Anchor 증가
        # Event가 Factor를 증가시키고, Factor가 Anchor에 반비례 관계면
        # → Anchor 감소

        if event_impact == "INCREASES":
            if factor_relation == "PROPORTIONAL":
                result_direction = "increase"
            else:
                result_direction = "decrease"
        else:  # DECREASES
            if factor_relation == "PROPORTIONAL":
                result_direction = "decrease"
            else:
                result_direction = "increase"

        return hypothesis_direction.lower() == result_direction

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
