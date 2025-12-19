"""
Analysis Agent - 분석 조율 에이전트 (Orchestrator)

역할:
- 가설 생성 → 가설 검증 (SQL) → 이벤트 매칭 플로우 조율
- 하위 에이전트들의 협업 관리
- 최종 분석 결과 종합 (SQL 쿼리 + 매칭된 이벤트 포함)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..base import BaseAgent, AgentContext
from .hypothesis_generator import HypothesisGenerator, Hypothesis
from .hypothesis_validator import HypothesisValidator
from .event_matcher import EventMatcher, MatchedEvent


@dataclass
class KPIChange:
    """KPI 변동 정보"""
    kpi_name: str  # 매출, 원가, 판매수량
    previous_value: float
    current_value: float
    change_percent: float
    change_amount: float
    period_info: str  # "2024 Q4 vs 2023 Q4"
    region: str = ""
    sql_query: str = ""


@dataclass
class AnalysisResult:
    """분석 결과"""
    question: str
    kpi_change: KPIChange = None  # KPI 변동 (먼저 보여줌)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    validated_hypotheses: List[Hypothesis] = field(default_factory=list)
    matched_events: Dict[str, List[MatchedEvent]] = field(default_factory=dict)
    sql_queries: List[Dict] = field(default_factory=list)
    summary: str = ""
    sources: List[Dict] = field(default_factory=list)
    details: List[Dict] = field(default_factory=list)


REASONING_PROMPT = """당신은 LG전자 HE(Home Entertainment) 사업부의 경영 전략 분석 전문가입니다.
경영진에게 보고할 **핵심 원인 {top_k}가지**를 분석하세요.

## 분석 질문
{question}

## KPI 변동 현황
{kpi_summary}

## 분석 데이터
{validated_hypotheses_detail}

---

## 작성 지침 (경영진 보고서 스타일)

### 1. 문체
- 경영 전략팀이 이해할 수 있는 **비즈니스 언어** 사용
- 기술적 용어 (Factor, Score, Graph, INCREASES 등) **절대 사용 금지**
- 자연스럽고 논리적인 문장으로 서술

### 2. 구조: 각 원인별 심층 분석
각 원인에 대해 **검증 유형에 따라** 다르게 설명:

#### [ERP 데이터 검증된 원인] (실적 데이터 기반)
**데이터 분석 결과**:
- 구체적 수치 변화 (예: "물류비가 전년 대비 15% 증가하여 원가 상승")
- 이 변화가 KPI에 미친 정량적 영향

**시장 환경 요인**: (관련 이벤트가 있는 경우)
- 관련 시장 동향, 출처 인용 [1], [2] 형식

#### [Knowledge Graph 기반 원인] (외부 요인, ERP 데이터 없음)
**인과관계 분석**:
- 제공된 인과관계 경로를 자연어로 설명
- 예: "홍해 사태로 인한 해상운임 상승이 물류비 증가로 이어져 원가 상승 압력"
- **주의**: 구체적 수치 변화는 언급하지 말 것 (ERP에 해당 데이터 없음)

**시장 환경 요인**:
- 관련 시장 동향, 출처 인용 [1], [2] 형식

### 3. 사업 영향 (근거가 있는 경우만)
- ERP 검증: 수치 변화가 전체 KPI에서 차지하는 비중으로 영향 설명
- Graph 검증: 인과관계 경로에서 도출된 영향만 설명
- **근거 없이 추측하지 말 것**

### 4. 분량
- 각 원인당 **150-250자** 상세 설명
- 총 분석 분량: 600-900자

### 5. 정확성
- 제공된 데이터와 뉴스만 인용 (새로운 수치 생성 금지)
- Graph 기반 원인은 "~로 분석됨", "~에 기인한 것으로 판단됨" 등으로 표현
- ERP에 없는 외부 요인(환율, 경쟁, 수요 등)은 수치 변화를 언급하지 않음

### 6. 결론
마지막에 **종합 분석** (2-3문장):
- 핵심 원인들의 복합 작용
- 경영 전략적 시사점

## 응답
"""


class AnalysisAgent(BaseAgent):
    """분석 조율 에이전트"""

    name = "analysis_agent"
    description = "가설 생성, SQL 검증, 이벤트 매칭을 조율하여 KPI 변동 원인을 분석합니다."

    # KPI 추출 패턴 (실제 DB 스키마에 맞춤)
    KPI_PATTERNS = {
        "매출": {
            "keywords": ["매출", "revenue", "sales", "수익"],
            "query_template": """
                SELECT
                    CASE
                        WHEN DOC_DATE >= '{prev_start}' AND DOC_DATE <= '{prev_end}' THEN 'Previous'
                        WHEN DOC_DATE >= '{curr_start}' AND DOC_DATE <= '{curr_end}' THEN 'Current'
                    END AS PERIOD,
                    SUM(si.NET_VALUE) AS TOTAL_VALUE
                FROM TBL_TX_SALES_HEADER sh
                JOIN TBL_TX_SALES_ITEM si ON sh.ORDER_NO = si.ORDER_NO
                WHERE (
                    (DOC_DATE >= '{prev_start}' AND DOC_DATE <= '{prev_end}')
                    OR (DOC_DATE >= '{curr_start}' AND DOC_DATE <= '{curr_end}')
                ) {region_filter}
                GROUP BY PERIOD
            """
        },
        "원가": {
            "keywords": ["원가", "cost", "비용"],
            "query_template": """
                SELECT
                    CASE
                        WHEN sh.DOC_DATE >= '{prev_start}' AND sh.DOC_DATE <= '{prev_end}' THEN 'Previous'
                        WHEN sh.DOC_DATE >= '{curr_start}' AND sh.DOC_DATE <= '{curr_end}' THEN 'Current'
                    END AS PERIOD,
                    SUM(cd.COST_AMOUNT) AS TOTAL_VALUE
                FROM TBL_TX_SALES_HEADER sh
                JOIN TBL_TX_COST_DETAIL cd ON sh.ORDER_NO = cd.ORDER_NO
                WHERE (
                    (sh.DOC_DATE >= '{prev_start}' AND sh.DOC_DATE <= '{prev_end}')
                    OR (sh.DOC_DATE >= '{curr_start}' AND sh.DOC_DATE <= '{curr_end}')
                ) {region_filter}
                GROUP BY PERIOD
            """
        },
        "판매수량": {
            "keywords": ["판매량", "수량", "quantity", "volume"],
            "query_template": """
                SELECT
                    CASE
                        WHEN DOC_DATE >= '{prev_start}' AND DOC_DATE <= '{prev_end}' THEN 'Previous'
                        WHEN DOC_DATE >= '{curr_start}' AND DOC_DATE <= '{curr_end}' THEN 'Current'
                    END AS PERIOD,
                    SUM(si.ORDER_QTY) AS TOTAL_VALUE
                FROM TBL_TX_SALES_HEADER sh
                JOIN TBL_TX_SALES_ITEM si ON sh.ORDER_NO = si.ORDER_NO
                WHERE (
                    (DOC_DATE >= '{prev_start}' AND DOC_DATE <= '{prev_end}')
                    OR (DOC_DATE >= '{curr_start}' AND DOC_DATE <= '{curr_end}')
                ) {region_filter}
                GROUP BY PERIOD
            """
        }
    }

    # 지역 → Subsidiary 매핑
    REGION_SUBSIDIARY_MAP = {
        "NA": ["LGEUS", "LGECA"],
        "EU": ["LGEDE", "LGEFR", "LGEUK"],
        "KR": ["LGEKR"],
        "US": ["LGEUS"],
        "북미": ["LGEUS", "LGECA"],
        "유럽": ["LGEDE", "LGEFR", "LGEUK"],
        "한국": ["LGEKR"]
    }

    # 유사 Factor 그룹화 (대표 Factor → 유사 Factor 목록)
    FACTOR_GROUPS = {
        # 수요 관련
        "수요 변동": ["수요", "글로벌수요", "지역별수요", "계절적 수요", "계절적수요", "IT 세트 수요 둔화",
                    "수요부진", "수요 부진", "TV수요", "가전수요", "성수기효과", "성수기 효과"],
        # 경기/소비 관련
        "경기/소비심리": ["경기부진", "경기 부진", "소비심리위축", "소비심리 위축", "소비 심리",
                      "침체된 주택 매매", "주택 매매", "경기침체", "소비 둔화"],
        # 환율 관련
        "환율": ["환율", "원/달러 환율", "달러 환율", "원달러", "달러 강세"],
        # 경쟁 관련
        "경쟁 심화": ["경쟁심화", "경쟁 심화", "가격경쟁", "중국업체 경쟁", "TCL", "하이센스"],
        # 물류/운임 관련
        "물류비/운임": ["물류비", "해상운임", "운임", "컨테이너 운임", "홍해 사태"],
        # 패널/부품 관련
        "패널/부품 가격": ["패널가격", "패널 가격", "디스플레이 가격", "OLED 패널", "LCD 패널", "부품비"],
        # 관세 관련
        "관세/무역": ["관세", "관세율", "수입관세", "트럼프 관세", "무역분쟁"],
    }

    # 분석 설정
    TOP_K_FACTORS = 3  # 상위 몇 개 원인만 상세 분석
    MIN_EVENT_SCORE = 0.5  # 이벤트 최소 매칭 점수
    REASONING_MODEL = "gpt-4o"  # 추론 모델: o1, o1-mini, gpt-4o (o1 미지원 시 gpt-4o 사용)

    def __init__(self, api_key: str = None, db_path: str = None):
        super().__init__(api_key)
        self.db_path = db_path

        # 하위 에이전트 초기화
        self.hypothesis_generator = HypothesisGenerator(api_key)
        self.hypothesis_validator = HypothesisValidator(api_key, db_path)
        self.event_matcher = EventMatcher(api_key)

        self.add_sub_agent(self.hypothesis_generator)
        self.add_sub_agent(self.hypothesis_validator)
        self.add_sub_agent(self.event_matcher)

    def analyze(
        self,
        question: str,
        period: Dict = None,
        region: str = None,
        company: str = "LGE",
        verbose: bool = True
    ) -> AnalysisResult:
        """
        KPI 변동 원인 분석 실행

        Args:
            question: 사용자 질문
            period: {"year": 2024, "quarter": 4}
            region: "NA", "EU", "KR" 등
            company: 회사 코드
            verbose: 상세 출력 여부
        """
        if verbose:
            print("=" * 60)
            print(f"질문: {question}")
            print("=" * 60)

        # 기본 기간 설정
        if not period:
            period = {"year": 2024, "quarter": 4}

        # Step 0: KPI 변동 계산 (매출/원가/수량 자체의 변동)
        if verbose:
            print("\n[Step 0] KPI 변동 계산 중...")

        kpi_change = self._calculate_kpi_change(question, period, region)

        if verbose and kpi_change:
            print(f"  {kpi_change.kpi_name}: {kpi_change.previous_value:,.0f} → {kpi_change.current_value:,.0f} ({kpi_change.change_percent:+.1f}%)")
            print(f"  비교 기간: {kpi_change.period_info}")

        # Step 1: 가설 생성
        if verbose:
            print("\n[Step 1] 가설 생성 중...")

        hypotheses = self.hypothesis_generator.generate(
            question=question,
            company=company,
            period=f"{period['year']}년 Q{period['quarter']}",
            region=region
        )

        if verbose:
            print(f"  생성된 가설: {len(hypotheses)}개")
            for h in hypotheses:
                print(f"    - [{h.id}] {h.description}")

        # Step 2: 가설 검증 (SQL Agent)
        if verbose:
            print("\n[Step 2] 가설 검증 중 (SQL Agent)...")

        validated = self.hypothesis_validator.validate(
            hypotheses=hypotheses,
            period=period,
            region=region,
            threshold=5.0
        )

        # SQL 쿼리 수집
        sql_queries = []
        if verbose:
            print(f"  검증된 가설: {len(validated)}개")
            for h in validated:
                data = h.validation_data or {}
                print(f"    - [{h.id}] {h.factor}: {data.get('details', '')}")

                # SQL 쿼리 저장 및 출력
                sql_query = data.get("sql_query", "")
                if sql_query:
                    sql_queries.append({
                        "hypothesis_id": h.id,
                        "factor": h.factor,
                        "sql": sql_query
                    })
                    print(f"      SQL: {sql_query[:100]}...")

        # Step 3: 이벤트 매칭 (Scoring Algorithm)
        if verbose:
            print("\n[Step 3] 이벤트 매칭 중 (Scoring Algorithm)...")

        matched_events = {}
        try:
            matched_events = self.event_matcher.match(
                hypotheses=validated,
                region=region,
                min_score=0.3,  # 0-1 스케일
                top_k=5
            )

            if verbose:
                for h_id, events in matched_events.items():
                    print(f"  [{h_id}] 매칭된 이벤트: {len(events)}개")
                    for ev in events[:3]:
                        print(f"    - {ev.event_name} (Score: {ev.total_score:.1f})")
                        if ev.sources:
                            print(f"      출처: {ev.sources[0].get('title', '')[:50]}...")

        except Exception as e:
            if verbose:
                print(f"  이벤트 매칭 오류: {e}")

        # Step 4: 결과 종합
        if verbose:
            print("\n[Step 4] 결과 종합 중...")

        result = AnalysisResult(
            question=question,
            kpi_change=kpi_change,  # KPI 변동 정보 추가
            hypotheses=hypotheses,
            validated_hypotheses=validated,
            matched_events=matched_events,
            sql_queries=sql_queries
        )

        # 상세 분석 결과 구성
        result.details = self._build_details(validated, matched_events, sql_queries)

        # Step 5: 추론 기반 요약 생성 (출처 포함)
        if verbose:
            print("\n[Step 5] 추론 기반 답변 생성 중...")

        summary_result = self._generate_summary(question, result.details, kpi_change)
        result.summary = summary_result["summary"]
        result.sources = summary_result["sources"]

        if verbose:
            print(f"  출처 수: {len(result.sources)}개")
            print("\n" + "=" * 60)
            print("분석 완료!")
            print("=" * 60)

        return result

    def _build_details(
        self,
        validated: List[Hypothesis],
        matched_events: Dict[str, List[MatchedEvent]],
        sql_queries: List[Dict]
    ) -> List[Dict]:
        """상세 분석 결과 구성 (SQL/Graph 검증 타입 구분)"""
        details = []

        # SQL 쿼리를 hypothesis_id로 매핑
        sql_map = {q["hypothesis_id"]: q["sql"] for q in sql_queries}

        for hypothesis in validated:
            h_data = hypothesis.validation_data or {}

            # 검증 타입 확인 (sql 또는 graph)
            validation_type = h_data.get("validation_type", "sql")
            graph_evidence = h_data.get("graph_evidence", {})

            prev_val = h_data.get("previous_value", 0)
            curr_val = h_data.get("current_value", 0)
            change_pct = h_data.get("change_percent", 0)

            # 데이터 방향성 해석 (SQL 검증된 경우만)
            if validation_type == "sql" and (prev_val != 0 or curr_val != 0):
                # 음수값: 비용/손실 → 값이 커지면(덜 음수) 개선, 작아지면(더 음수) 악화
                # 양수값: 매출/이익 → 값이 커지면 개선, 작아지면 악화
                if prev_val < 0 and curr_val < 0:
                    if curr_val > prev_val:
                        interpretation = "개선 (손실/비용 감소)"
                        impact_direction = "positive"
                    else:
                        interpretation = "악화 (손실/비용 증가)"
                        impact_direction = "negative"
                elif prev_val >= 0 and curr_val >= 0:
                    if curr_val > prev_val:
                        interpretation = "증가"
                        impact_direction = "positive"
                    else:
                        interpretation = "감소"
                        impact_direction = "negative"
                else:
                    if curr_val > prev_val:
                        interpretation = "개선 (적자→흑자 또는 손실 감소)"
                        impact_direction = "positive"
                    else:
                        interpretation = "악화 (흑자→적자 또는 손실 증가)"
                        impact_direction = "negative"
            else:
                # Graph 검증인 경우: 인과관계 경로에서 해석
                interpretation = h_data.get("details", hypothesis.description)
                impact_direction = hypothesis.direction  # increase/decrease

            # 상세 결과 구성
            detail = {
                "factor": hypothesis.factor,
                "category": hypothesis.category,
                "description": hypothesis.description,
                "validation_type": validation_type,  # "sql" or "graph"
                "change_percent": change_pct,
                "previous_value": prev_val,
                "current_value": curr_val,
                "direction": h_data.get("direction", hypothesis.direction),
                "interpretation": interpretation,
                "impact_direction": impact_direction,
                "sql_query": sql_map.get(hypothesis.id, "") if validation_type == "sql" else "",
                "matched_events": [],
                # Graph 검증 시 인과관계 경로 포함
                "graph_evidence": graph_evidence if validation_type == "graph" else {},
                "causal_chains": graph_evidence.get("causal_chains", []) if validation_type == "graph" else []
            }

            # 매칭된 이벤트 추가 (Scoring Algorithm 결과)
            events = matched_events.get(hypothesis.id, [])
            for ev in events[:5]:
                detail["matched_events"].append({
                    "name": ev.event_name,
                    "category": ev.event_category,
                    "severity": ev.severity,
                    "impact": ev.impact_type,
                    "score": ev.total_score,
                    "score_breakdown": ev.score_breakdown,
                    "sources": ev.sources[:2],
                    "evidence": ev.evidence[:200] if ev.evidence else ""
                })

            details.append(detail)

        # 정렬: SQL 검증(수치 있음)은 변화율 순, Graph 검증은 이벤트 수 순
        def sort_key(d):
            if d["validation_type"] == "sql" and d["change_percent"] != 0:
                return (0, abs(d["change_percent"]))  # SQL 검증 우선, 변화율 순
            else:
                return (1, len(d.get("matched_events", [])))  # Graph는 이벤트 수 순

        details.sort(key=sort_key, reverse=True)

        return details

    def _get_representative_factor(self, factor_name: str) -> str:
        """Factor의 대표 그룹명 반환"""
        factor_lower = factor_name.lower().strip()
        for group_name, members in self.FACTOR_GROUPS.items():
            for member in members:
                if member.lower() in factor_lower or factor_lower in member.lower():
                    return group_name
        return factor_name  # 그룹에 없으면 원래 이름 반환

    def _select_top_factors(
        self,
        details: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """
        유사 Factor 그룹화 후 Top K 선정

        선정 기준:
        1. 그룹별 대표 Factor 선정 (가장 높은 변화율)
        2. 이벤트 매칭 품질 (고품질 이벤트가 있는 Factor 우선)
        3. 변화율 크기 순 정렬
        """
        if top_k is None:
            top_k = self.TOP_K_FACTORS

        if not details:
            return []

        # 1. 그룹별로 Factor 분류
        group_map = {}  # group_name -> [details]
        for d in details:
            factor = d["factor"]
            group = self._get_representative_factor(factor)
            if group not in group_map:
                group_map[group] = []
            group_map[group].append(d)

        # 2. 각 그룹에서 대표 Factor 선정 (변화율 + 이벤트 품질)
        representatives = []
        for group_name, group_details in group_map.items():
            # 그룹 내 정렬: 이벤트 품질 → 변화율
            def score_detail(d):
                change_score = abs(d["change_percent"])
                # 고품질 이벤트 보너스 (score >= MIN_EVENT_SCORE)
                high_quality_events = [
                    e for e in d.get("matched_events", [])
                    if e.get("score", 0) >= self.MIN_EVENT_SCORE
                ]
                event_bonus = len(high_quality_events) * 10
                return change_score + event_bonus

            group_details.sort(key=score_detail, reverse=True)
            best = group_details[0]

            # 그룹 정보 추가
            best["group_name"] = group_name
            best["group_size"] = len(group_details)
            if len(group_details) > 1:
                best["related_factors"] = [d["factor"] for d in group_details[1:]]
            else:
                best["related_factors"] = []

            representatives.append(best)

        # 3. 대표 Factor들 중 Top K 선정
        def final_score(d):
            change_score = abs(d["change_percent"])
            high_quality_events = [
                e for e in d.get("matched_events", [])
                if e.get("score", 0) >= self.MIN_EVENT_SCORE
            ]
            event_bonus = len(high_quality_events) * 15
            return change_score + event_bonus

        representatives.sort(key=final_score, reverse=True)

        return representatives[:top_k]

    def _generate_summary(
        self,
        question: str,
        details: List[Dict],
        kpi_change: KPIChange = None
    ) -> Dict[str, Any]:
        """추론 모델 기반 분석 요약 생성 (Top K 핵심 원인 심층 분석)"""
        if not details and not kpi_change:
            return {
                "summary": "검증된 원인을 찾지 못했습니다.",
                "sources": []
            }

        # 0. KPI 변동 현황 포맷팅
        if kpi_change:
            change_direction = "증가" if kpi_change.change_percent > 0 else "감소"
            kpi_summary = f"""**{kpi_change.kpi_name}** 변동:
- 기간: {kpi_change.period_info}
- 이전 기간: {kpi_change.previous_value:,.0f}
- 현재 기간: {kpi_change.current_value:,.0f}
- 변화율: **{kpi_change.change_percent:+.1f}%** ({change_direction})
- 변화 금액: {kpi_change.change_amount:+,.0f}
"""
        else:
            kpi_summary = "(KPI 변동 정보 없음)"

        # 1. Top K Factor 선정 (유사 Factor 그룹화 후)
        print(f"[AnalysisAgent] 전체 검증된 가설: {len(details)}개")
        top_factors = self._select_top_factors(details, self.TOP_K_FACTORS)
        top_k = len(top_factors)
        print(f"[AnalysisAgent] Top {self.TOP_K_FACTORS} 선정 결과: {top_k}개")

        # Top Factor가 없으면 원본 details 사용 (최대 3개)
        if not top_factors and details:
            print("[AnalysisAgent] Top Factor 선정 실패, 원본 데이터 사용")
            top_factors = details[:self.TOP_K_FACTORS]
            top_k = len(top_factors)

        # 2. 선정된 Factor별 상세 정보 구성
        all_sources = []
        source_idx = 1
        validated_hypotheses_detail = ""

        if top_factors:
            for i, d in enumerate(top_factors, 1):
                factor = d['factor']
                category = d['category']
                change_pct = d['change_percent']
                prev_val = d['previous_value']
                curr_val = d['current_value']
                interpretation = d.get('interpretation', d.get('direction', ''))
                validation_type = d.get('validation_type', 'sql')
                causal_chains = d.get('causal_chains', [])

                # 그룹 정보
                group_name = d.get('group_name', factor)

                # 카테고리 한글화
                category_kr = {
                    "cost": "원가 요인",
                    "revenue": "매출 요인",
                    "pricing": "가격 요인",
                    "external": "외부 환경"
                }.get(category, category)

                # 검증 타입에 따라 다른 형식으로 출력
                if validation_type == "sql" and (prev_val != 0 or curr_val != 0):
                    # SQL 검증: 실적 데이터 기반
                    validated_hypotheses_detail += f"""
### 원인 {i}: {group_name}
**분류:** {category_kr}
**검증 방식:** ERP 실적 데이터

**실적 데이터 변화:**
- 변화율: {change_pct:+.1f}%
- 전년 동기: {prev_val:,.0f}
- 당기: {curr_val:,.0f}
- 해석: {interpretation}
"""
                else:
                    # Graph 검증: 인과관계 경로 기반
                    validated_hypotheses_detail += f"""
### 원인 {i}: {group_name}
**분류:** {category_kr}
**검증 방식:** Knowledge Graph 인과관계 분석 (ERP에 해당 데이터 없음)

**인과관계 경로:**
"""
                    # 인과관계 경로 출력
                    if causal_chains:
                        for chain in causal_chains[:3]:
                            chain_text = chain.get('chain_text', '')
                            if chain_text:
                                validated_hypotheses_detail += f"- {chain_text}\n"
                    else:
                        validated_hypotheses_detail += f"- {interpretation}\n"

                    validated_hypotheses_detail += """
**주의:** 이 요인은 ERP에 직접적인 수치 데이터가 없어 정량적 영향을 산출할 수 없습니다.
아래 시장 동향을 바탕으로 정성적 분석을 제공합니다.
"""

                # 외부 이벤트 추가 (고품질 이벤트만, 비즈니스 언어로)
                matched_events = d.get('matched_events', [])
                high_quality_events = [
                    e for e in matched_events
                    if e.get('score', 0) >= self.MIN_EVENT_SCORE
                ]

                if high_quality_events:
                    validated_hypotheses_detail += f"\n**관련 시장 동향:**\n"

                    for ev in high_quality_events[:3]:
                        event_name = ev.get('name', '')
                        evidence = ev.get('evidence', '')

                        # 출처 수집
                        event_source_refs = []
                        for src in ev.get('sources', [])[:2]:
                            title = src.get('title', '제목 없음')
                            url = src.get('link', src.get('url', ''))
                            if url:
                                existing = next((s for s in all_sources if s['url'] == url), None)
                                if existing:
                                    event_source_refs.append(f"[{existing['idx']}]")
                                else:
                                    all_sources.append({
                                        "idx": source_idx,
                                        "title": title,
                                        "url": url,
                                        "event": event_name,
                                        "factor": factor
                                    })
                                    event_source_refs.append(f"[{source_idx}]")
                                    source_idx += 1

                        source_str = " ".join(event_source_refs)

                        # 비즈니스 친화적 포맷 (기술 용어 제거)
                        validated_hypotheses_detail += f"""
- **{event_name}** {source_str}
  {evidence[:400] if evidence else ''}
"""
                else:
                    if validation_type == "sql":
                        validated_hypotheses_detail += "\n**관련 시장 동향:** 직접 관련된 외부 이슈가 확인되지 않음 (내부 실적 데이터 기반 분석)\n"
                    else:
                        validated_hypotheses_detail += "\n**관련 시장 동향:** 관련 뉴스/이벤트가 확인되지 않음\n"

        else:
            validated_hypotheses_detail = "(분석 가능한 데이터 없음)"

        # 출처 목록 추가
        if all_sources:
            validated_hypotheses_detail += "\n---\n**📚 참고 출처:**\n"
            for src in all_sources:
                validated_hypotheses_detail += f"[{src['idx']}] {src['title']}\n"

        # 3. 추론 프롬프트 구성
        prompt = REASONING_PROMPT.format(
            question=question,
            kpi_summary=kpi_summary,
            validated_hypotheses_detail=validated_hypotheses_detail,
            top_k=top_k
        )

        # 4. 추론 모델 호출 (o1 → gpt-4o fallback)
        system_prompt = """당신은 LG전자 경영 전략실 소속 재무/시장 분석 전문가입니다.

작성 원칙:
1. 경영진이 바로 이해할 수 있는 비즈니스 언어만 사용
2. 기술 용어 절대 금지: Factor, Score, Graph, INCREASES, DECREASES, KPI 등
3. 자연스럽고 논리적인 문장으로 흐름있게 서술
4. 실적 수치는 정확히 인용하되, "전년 대비 24% 감소" 등 자연어로 표현
5. 시장 동향은 구체적 사례와 출처 번호 [1], [2]로 인용"""

        summary = None

        # 1차 시도: 설정된 추론 모델
        try:
            print(f"[AnalysisAgent] 추론 모델 호출: {self.REASONING_MODEL}")
            summary = self._call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.REASONING_MODEL,
                temperature=0.2,
                max_tokens=3000
            )
        except Exception as e:
            print(f"[AnalysisAgent] {self.REASONING_MODEL} 호출 실패: {e}")

        # 2차 시도: fallback to gpt-4o
        if not summary:
            try:
                print("[AnalysisAgent] Fallback: gpt-4o 사용")
                summary = self._call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model="gpt-4o",
                    temperature=0.3,
                    max_tokens=2500
                )
            except Exception as e:
                print(f"[AnalysisAgent] gpt-4o 호출 실패: {e}")
                return {
                    "summary": f"분석 생성 오류: {e}",
                    "sources": all_sources
                }

        # 5. 출처 섹션 항상 추가
        if all_sources and summary:
            # 기존 출처 섹션이 있으면 제거
            if "### 출처" in summary:
                summary = summary.split("### 출처")[0].strip()

            summary += "\n\n---\n**출처:**\n"
            for src in all_sources[:10]:
                summary += f"- [{src['idx']}] [{src['title']}]({src['url']})\n"

        return {
            "summary": summary or "분석 결과를 생성하지 못했습니다.",
            "sources": all_sources
        }

    def run(self, context: AgentContext) -> Dict[str, Any]:
        """Agent 실행"""
        question = context.query
        metadata = context.metadata or {}

        result = self.analyze(
            question=question,
            period=metadata.get("period", {"year": 2024, "quarter": 4}),
            region=metadata.get("region"),
            company=metadata.get("company", "LGE"),
            verbose=metadata.get("verbose", True)
        )

        return {
            "question": result.question,
            "kpi_change": {
                "kpi_name": result.kpi_change.kpi_name if result.kpi_change else None,
                "change_percent": result.kpi_change.change_percent if result.kpi_change else None,
                "previous_value": result.kpi_change.previous_value if result.kpi_change else None,
                "current_value": result.kpi_change.current_value if result.kpi_change else None,
            } if result.kpi_change else None,
            "hypotheses_count": len(result.hypotheses),
            "validated_count": len(result.validated_hypotheses),
            "sql_queries": result.sql_queries,
            "matched_events_count": sum(len(v) for v in result.matched_events.values()),
            "summary": result.summary,
            "sources": result.sources,
            "details": result.details
        }

    def _calculate_kpi_change(
        self,
        question: str,
        period: Dict,
        region: str = None
    ) -> Optional[KPIChange]:
        """
        질문에서 KPI 추출 후 변동 계산

        Args:
            question: 사용자 질문
            period: {"year": 2024, "quarter": 4}
            region: 지역 코드

        Returns:
            KPIChange 또는 None
        """
        # 1. 질문에서 KPI 추출
        kpi_name = self._extract_kpi_from_question(question)
        kpi_info = self.KPI_PATTERNS.get(kpi_name)

        if not kpi_info:
            return None

        # 2. 기간 계산 (DATE 형식: YYYY-MM-DD)
        year = period.get("year", 2024)
        quarter = period.get("quarter", 4)

        curr_start, curr_end = self._get_quarter_date_range(year, quarter)
        prev_start, prev_end = self._get_quarter_date_range(year - 1, quarter)  # 전년 동기

        # 3. 지역 필터 생성 (SUBSIDIARY_ID 기반)
        region_filter = ""
        if region:
            subsidiaries = self.REGION_SUBSIDIARY_MAP.get(region.upper(), [])
            if not subsidiaries:
                subsidiaries = self.REGION_SUBSIDIARY_MAP.get(region, [])
            if subsidiaries:
                subs_str = ", ".join([f"'{s}'" for s in subsidiaries])
                region_filter = f"AND sh.SUBSIDIARY_ID IN ({subs_str})"

        # 4. SQL 쿼리 생성 (템플릿 사용)
        sql_query = kpi_info["query_template"].format(
            prev_start=prev_start,
            prev_end=prev_end,
            curr_start=curr_start,
            curr_end=curr_end,
            region_filter=region_filter
        )

        # 5. SQL 실행
        try:
            exec_result = self.hypothesis_validator.sql_executor.execute(sql_query)

            if not exec_result.success or exec_result.data is None:
                print(f"KPI 계산 SQL 실행 실패: {exec_result.error}")
                print(f"SQL: {sql_query}")
                return None

            data = exec_result.data.to_dict('records')

            prev_row = next((r for r in data if r.get('PERIOD') == 'Previous'), None)
            curr_row = next((r for r in data if r.get('PERIOD') == 'Current'), None)

            if not prev_row or not curr_row:
                print(f"KPI 데이터 없음: prev={prev_row}, curr={curr_row}")
                return None

            prev_value = float(prev_row.get('TOTAL_VALUE', 0) or 0)
            curr_value = float(curr_row.get('TOTAL_VALUE', 0) or 0)

            if prev_value == 0:
                change_percent = 100.0 if curr_value > 0 else 0.0
            else:
                change_percent = ((curr_value - prev_value) / abs(prev_value)) * 100

            change_amount = curr_value - prev_value

            region_text = region.upper() if region else "전체"
            period_info = f"{year}년 Q{quarter} vs {year-1}년 Q{quarter} ({region_text})"

            return KPIChange(
                kpi_name=kpi_name,
                previous_value=prev_value,
                current_value=curr_value,
                change_percent=round(change_percent, 1),
                change_amount=change_amount,
                period_info=period_info,
                region=region or "",
                sql_query=sql_query
            )

        except Exception as e:
            print(f"KPI 계산 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_kpi_from_question(self, question: str) -> str:
        """질문에서 KPI 추출"""
        question_lower = question.lower()

        for kpi_name, info in self.KPI_PATTERNS.items():
            for keyword in info["keywords"]:
                if keyword in question_lower:
                    return kpi_name

        # 기본값: 매출
        return "매출"

    def _get_quarter_range(self, year: int, quarter: int) -> tuple:
        """분기 시작/종료 월 계산 (YEARMONTH 형식)"""
        quarter_months = {
            1: ("01", "03"),
            2: ("04", "06"),
            3: ("07", "09"),
            4: ("10", "12")
        }
        start_month, end_month = quarter_months[quarter]
        return f"{year}-{start_month}", f"{year}-{end_month}"

    def _get_quarter_date_range(self, year: int, quarter: int) -> tuple:
        """분기 시작/종료 날짜 계산 (DATE 형식: YYYY-MM-DD)"""
        quarter_dates = {
            1: ("01-01", "03-31"),
            2: ("04-01", "06-30"),
            3: ("07-01", "09-30"),
            4: ("10-01", "12-31")
        }
        start_date, end_date = quarter_dates[quarter]
        return f"{year}-{start_date}", f"{year}-{end_date}"
