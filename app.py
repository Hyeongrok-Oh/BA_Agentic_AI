"""
LG Electronics HE Business Intelligence - Multi-Agent System UI
"""

import streamlit as st
import sys
import os
import json
import time
from datetime import datetime

# 경로 설정 (Docker 및 로컬 환경 모두 지원)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'intent_classifier/src'))

# .env 로드 (python-dotenv 사용 가능하면 사용, 아니면 수동 로드)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except ImportError:
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Import agents
from agents import Orchestrator
from agents.base import AgentContext
from agents.analysis import AnalysisAgent
from agents.search_agent import SearchAgent

# Intent Classifier import (팀원이 만든 것)
try:
    from intent_classifier import IntentClassifier
    INTENT_CLASSIFIER_AVAILABLE = True
except ImportError:
    INTENT_CLASSIFIER_AVAILABLE = False


# 페이지 설정
st.set_page_config(
    page_title="LG HE BI System",
    page_icon="📊",
    layout="wide"
)

# 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #A50034;
        margin-bottom: 1rem;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """세션 상태 초기화"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None


def classify_intent(query: str) -> dict:
    """Intent 분류"""
    if INTENT_CLASSIFIER_AVAILABLE:
        try:
            classifier = IntentClassifier()
            result = classifier.classify(query)
            return {
                "service_type": result.get("intent", "data_qa"),
                "analysis_mode": result.get("analysis_mode", "descriptive"),
                "sub_intent": result.get("sub_intent", "internal_data"),
                "query": query,
                "extracted_entities": result.get("extracted_entities", {}),
                "thinking": result.get("thinking", ""),
                "raw_result": result
            }
        except Exception as e:
            st.warning(f"Intent Classifier 오류: {e}")

    # Fallback: Orchestrator의 간단한 분류 사용
    return st.session_state.orchestrator._simple_classify(query)


def display_intent_result(intent_result: dict):
    """Intent 분류 결과 표시"""
    st.markdown("### 🎯 Step 1: Intent Classification")

    col1, col2, col3 = st.columns(3)

    with col1:
        service = intent_result.get("service_type", "data_qa")
        if service == "report_generation":
            st.metric("서비스 유형", "📄 Report Generation")
        else:
            st.metric("서비스 유형", "💬 Data Q&A")

    with col2:
        mode = intent_result.get("analysis_mode", "descriptive")
        if mode == "diagnostic":
            st.metric("분석 모드", "🔍 Diagnostic (원인 분석)")
        else:
            st.metric("분석 모드", "📊 Descriptive (데이터 조회)")

    with col3:
        sub = intent_result.get("sub_intent", "internal_data")
        if sub == "external_data":
            st.metric("데이터 소스", "🌐 External (Graph)")
        elif sub == "hybrid":
            st.metric("데이터 소스", "🔄 Hybrid")
        else:
            st.metric("데이터 소스", "🏢 Internal (ERP)")

    # 추출된 엔티티
    entities = intent_result.get("extracted_entities", {})
    if entities:
        with st.expander("📋 추출된 엔티티", expanded=False):
            st.json(entities)

    # Thinking (있으면)
    thinking = intent_result.get("thinking", "")
    if thinking:
        with st.expander("💭 Intent 분석 과정", expanded=False):
            st.write(thinking)


def display_hypothesis_generation(hypotheses: list):
    """가설 생성 결과 표시 (Graph-Based 상세 정보 포함)"""
    st.markdown("### 💡 Step 2: Hypothesis Generation (Graph-Based)")

    # Graph 기반 가설 수 계산
    graph_based = sum(1 for h in hypotheses if h.graph_evidence.get("from_graph", False))
    st.info(f"생성된 가설: **{len(hypotheses)}개** (Graph 기반: {graph_based}개)")

    for h in hypotheses:
        # Graph 기반 여부에 따른 아이콘
        evidence = h.graph_evidence or {}
        is_graph = evidence.get("from_graph", False)
        graph_icon = "🔗" if is_graph else "💭"

        # 카테고리별 색상
        category_colors = {
            "cost": "🔴", "revenue": "🟢", "pricing": "🔵", "external": "🟡"
        }
        cat_icon = category_colors.get(h.category, "⚪")

        with st.expander(f"{graph_icon} [{h.id}] {h.factor} {cat_icon}", expanded=False):
            # 인과관계 체인 (있으면)
            if hasattr(h, 'reasoning') and h.reasoning:
                st.markdown(f"**🔄 인과관계:** `{h.reasoning}`")
                st.markdown("---")

            # 상세 설명 (Markdown)
            st.markdown(h.description)

            # Graph Evidence 표시
            if is_graph:
                st.markdown("---")
                st.markdown("**📊 Knowledge Graph Evidence:**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    relation = evidence.get("relation_type", "N/A")
                    relation_kr = "동비례 ↑↑" if relation == "PROPORTIONAL" else "역비례 ↑↓"
                    st.metric("관계 유형", relation_kr)
                with col2:
                    mention = evidence.get("mention_count", 0)
                    st.metric("언급 횟수", f"{mention}회")
                with col3:
                    event_count = evidence.get("event_count", 0)
                    st.metric("관련 이벤트", f"{event_count}개")

            # 관련 이벤트 상세
            if hasattr(h, 'related_events') and h.related_events:
                st.markdown("---")
                st.markdown(f"**🔔 관련 이벤트 ({len(h.related_events)}건):**")

                for ev in h.related_events[:3]:
                    severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(ev.severity, "⚪")
                    impact_text = "증가" if ev.impact_direction == "INCREASES" else "감소"
                    regions = ", ".join([r for r in ev.target_regions if r]) if ev.target_regions else "전체"

                    st.markdown(f"""
                    {severity_emoji} **{ev.name}** ({ev.category})
                    - 영향: {h.factor} {impact_text} 유발
                    - 심각도: {ev.severity} | 지역: {regions}
                    """)
                    if ev.evidence:
                        st.caption(f"→ {ev.evidence[:150]}...")

            if h.sql_template:
                st.markdown("---")
                st.code(h.sql_template, language="sql")


def display_hypothesis_validation(validated: list, all_hypotheses: list):
    """가설 검증 결과 표시 (검증됨 + 기각됨 모두)"""
    st.markdown("### ✅ Step 3: Hypothesis Validation (SQL)")

    # 검증된 가설과 기각된 가설 분리
    validated_ids = {h.id for h in validated}
    rejected = [h for h in all_hypotheses if h.id not in validated_ids]

    st.success(f"검증된 가설: **{len(validated)}/{len(all_hypotheses)}개** | 기각된 가설: **{len(rejected)}개**")

    # 검증된 가설
    if validated:
        st.markdown("#### ✅ 검증된 가설 (Validated)")
        for h in validated:
            data = h.validation_data or {}
            change = data.get("change_percent", 0)

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**[{h.id}] {h.factor}**")
            with col2:
                st.metric(
                    "변화율",
                    f"{change:+.1f}%",
                    delta=f"{data.get('direction', '')}"
                )
            with col3:
                st.write(f"{data.get('previous_value', 0):,.0f} → {data.get('current_value', 0):,.0f}")

            # SQL 쿼리 표시
            sql_query = data.get('sql_query', '')
            if sql_query:
                with st.expander(f"🔍 SQL Query - {h.factor}", expanded=False):
                    st.code(sql_query, language="sql")

    # 기각된 가설 (expander로 닫혀있음)
    if rejected:
        with st.expander(f"❌ 기각된 가설 ({len(rejected)}개) - 클릭하여 상세 확인", expanded=False):
            for h in rejected:
                data = h.validation_data or {}
                change = data.get("change_percent", 0)
                direction = data.get("direction", "unknown")

                # 기각 사유 판단
                if data:
                    if abs(change) < 5.0:
                        reject_reason = f"변동률 미달 ({change:+.1f}% < ±5%)"
                    elif h.direction.lower() == "increase" and direction == "decreased":
                        reject_reason = f"방향 불일치 (예상: 증가, 실제: 감소 {change:+.1f}%)"
                    elif h.direction.lower() == "decrease" and direction == "increased":
                        reject_reason = f"방향 불일치 (예상: 감소, 실제: 증가 {change:+.1f}%)"
                    else:
                        reject_reason = f"기타 ({direction}, {change:+.1f}%)"
                else:
                    reject_reason = "데이터 없음 또는 SQL 오류"

                st.markdown(f"""
                <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ffc107;">
                    <strong>[{h.id}] {h.factor}</strong><br>
                    <span style="color: #856404;">기각 사유: {reject_reason}</span><br>
                    <span style="font-size: 0.9em;">가설: {h.description}</span>
                </div>
                """, unsafe_allow_html=True)

                # SQL 쿼리 표시 (있는 경우)
                sql_query = data.get('sql_query', '')
                if sql_query:
                    with st.expander(f"🔍 SQL Query - {h.factor}", expanded=False):
                        st.code(sql_query, language="sql")


def display_event_matching(matched_events: dict):
    """이벤트 매칭 결과 표시 (하이브리드 스코어링)"""
    st.markdown("### 🎯 Step 4: Event Matching (Hybrid Scoring)")

    total_events = sum(len(v) for v in matched_events.values())
    st.info(f"매칭된 이벤트: **{total_events}개** (Vector + Graph 하이브리드)")

    for h_id, events in matched_events.items():
        st.write(f"**[{h_id}]** - {len(events)}개 이벤트 매칭")

        for ev in events[:5]:
            # 스코어에 따른 색상 (0-1 스케일)
            score = ev.total_score
            if score >= 0.7:
                score_color = "🟢"
            elif score >= 0.4:
                score_color = "🟡"
            else:
                score_color = "🔴"

            with st.expander(f"{score_color} {ev.event_name} (Score: {score:.2f})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**카테고리**: {ev.event_category}")
                    st.write(f"**영향**: {ev.impact_type} → {ev.matched_factor}")
                    st.write(f"**심각도**: {ev.severity}")
                with col2:
                    st.write(f"**지역**: {', '.join(ev.target_regions) if ev.target_regions else '전체'}")
                    st.write(f"**Magnitude**: {ev.magnitude}")

                # Score Breakdown (하이브리드)
                st.write("**Score Breakdown:**")
                breakdown = ev.score_breakdown

                # Semantic vs Graph 비교
                col_sem, col_graph = st.columns(2)
                with col_sem:
                    semantic = breakdown.get('semantic', 0)
                    st.metric("🔍 Semantic (40%)", f"{semantic:.2f}", help="Vector Similarity")
                with col_graph:
                    graph = breakdown.get('graph', 0)
                    st.metric("🔗 Graph (60%)", f"{graph:.2f}", help="KG 관계 기반")

                # Graph 세부 점수
                st.caption("Graph Score 세부:")
                cols = st.columns(4)
                cols[0].write(f"Direction: {breakdown.get('direction', 0):.1f}")
                cols[1].write(f"Magnitude: {breakdown.get('magnitude', 0):.1f}")
                cols[2].write(f"Region: {breakdown.get('region', 0):.1f}")
                cols[3].write(f"Severity: {breakdown.get('severity', 0):.1f}")

                # 출처
                if ev.sources:
                    st.write("**출처:**")
                    for src in ev.sources[:2]:
                        title = src.get('title', 'N/A')
                        url = src.get('url', '')
                        if url:
                            st.markdown(f"- [{title[:60]}...]({url})")
                        else:
                            st.write(f"- {title[:60]}...")

                if ev.evidence:
                    st.write("**근거:**")
                    st.caption(ev.evidence[:300] + "..." if len(ev.evidence) > 300 else ev.evidence)


def display_evidence_collection(evidences: dict):
    """증거 수집 결과 표시 (레거시)"""
    st.markdown("### 🔗 Step 4: Evidence Collection (Graph)")

    total_events = sum(len(v) for v in evidences.values())
    st.info(f"발견된 관련 이벤트: **{total_events}개**")

    for h_id, ev_list in evidences.items():
        st.write(f"**[{h_id}]** - {len(ev_list)}개 이벤트")

        for ev in ev_list[:5]:
            with st.expander(f"📰 {ev.event_name} ({ev.event_category})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**영향**: {ev.impact_type} → {ev.factor_name}")
                    st.write(f"**심각도**: {ev.event_severity}")
                with col2:
                    st.write(f"**지역**: {', '.join(ev.target_regions) if ev.target_regions else '전체'}")

                if ev.evidence_text:
                    st.write("**근거:**")
                    st.caption(ev.evidence_text[:300] + "..." if len(ev.evidence_text) > 300 else ev.evidence_text)


def display_graph_query(evidences: dict):
    """Graph Query 표시"""
    if evidences:
        with st.expander("🔍 Cypher Query 예시", expanded=False):
            sample_query = """
MATCH (e:Event)-[r:INCREASES|DECREASES]->(f:Factor)
WHERE f.name CONTAINS $factor_name
OPTIONAL MATCH (e)-[:TARGETS]->(d:Dimension)
RETURN e.name, e.category, e.evidence,
       type(r) as impact, f.name as factor
ORDER BY e.severity DESC
LIMIT 10
"""
            st.code(sample_query, language="cypher")


def display_vector_search_results(events: list):
    """벡터 검색 결과 (이벤트 목록) 표시"""
    if not events:
        st.warning("관련 이벤트를 찾지 못했습니다.")
        return

    st.success(f"**{len(events)}개** 유사 이벤트 발견")

    for i, event in enumerate(events, 1):
        # 유사도 점수에 따른 색상
        score = event.get("score", 0)
        if score > 0.8:
            score_color = "🟢"
        elif score > 0.6:
            score_color = "🟡"
        else:
            score_color = "🔴"

        # 심각도 배지
        severity = event.get("severity", "medium")
        severity_badge = {"high": "🔴 높음", "medium": "🟡 보통", "low": "🟢 낮음"}.get(severity, "보통")

        # 카테고리 이모지
        category = event.get("category", "")
        category_emoji = {
            "geopolitical": "🌍",
            "policy": "📜",
            "market": "📈",
            "company": "🏢",
            "macro_economy": "💹",
            "technology": "🔬"
        }.get(category, "📰")

        with st.expander(f"{score_color} [{i}] {event.get('name', 'Unknown Event')} ({category_emoji} {category})", expanded=(i <= 2)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("유사도", f"{score:.2%}")
            with col2:
                st.write(f"**심각도**: {severity_badge}")
            with col3:
                st.write(f"**카테고리**: {category}")

            # 관련 Factor 표시
            related_factors = event.get("related_factors", [])
            if related_factors:
                st.write("**영향 Factor:**")
                st.write(", ".join([f"`{f}`" for f in related_factors[:5]]))

            # Evidence
            evidence = event.get("evidence", "")
            if evidence:
                st.write("**근거:**")
                st.caption(evidence[:500] + ("..." if len(evidence) > 500 else ""))

            # 출처 URL
            source_urls = event.get("source_urls", [])
            source_titles = event.get("source_titles", [])
            if source_urls:
                st.write("**출처:**")
                for j, url in enumerate(source_urls[:3]):
                    title = source_titles[j] if j < len(source_titles) else f"출처 {j+1}"
                    st.markdown(f"- [{title}]({url})")


def display_summary(summary_result: dict, details: list):
    """분석 결과 표시 (문장형 답변 + 출처)"""
    st.markdown("### 📝 Step 5: Analysis Result")

    # summary_result가 dict인 경우와 str인 경우 모두 처리
    if isinstance(summary_result, dict):
        summary = summary_result.get("summary", "")
        sources = summary_result.get("sources", [])
    else:
        summary = summary_result
        sources = []

    # 분석 결과 (문장형)
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #A50034; line-height: 1.8;">
    {summary.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


def run_analysis(query: str):
    """분석 실행 및 결과 표시"""

    # Progress bar
    progress = st.progress(0)
    status = st.empty()

    # Step 1: Intent Classification
    status.text("🎯 Intent 분류 중...")
    progress.progress(10)

    intent_result = classify_intent(query)
    time.sleep(0.3)

    # Intent 결과 표시
    display_intent_result(intent_result)
    progress.progress(20)

    st.markdown("---")

    # 분석 모드에 따른 처리
    analysis_mode = intent_result.get("analysis_mode", "descriptive")

    if analysis_mode == "diagnostic":
        # Diagnostic: Analysis Agent 사용
        # 1. 먼저 모든 분석 단계 실행 (데이터 수집)
        analysis_agent = AnalysisAgent()

        entities = intent_result.get("extracted_entities", {})
        period = entities.get("period", {"year": 2024, "quarter": 4})
        region = entities.get("region")
        if isinstance(region, list):
            region = region[0] if region else None

        # Step 1: KPI 변동 계산
        status.text("📊 KPI 변동 계산 중...")
        kpi_change = analysis_agent._calculate_kpi_change(query, period, region)
        progress.progress(20)

        # Step 2: 가설 생성
        status.text("💡 가설 생성 중...")
        hypotheses = analysis_agent.hypothesis_generator.generate(
            question=query,
            company=entities.get("company", "LGE"),
            period=f"{period.get('year', 2024)}년 Q{period.get('quarter', 4)}",
            region=region
        )
        progress.progress(35)

        # Step 3: 가설 검증
        status.text("✅ 가설 검증 중 (SQL Agent)...")
        validated = analysis_agent.hypothesis_validator.validate(
            hypotheses=hypotheses,
            period=period,
            region=region,
            threshold=5.0
        )

        # SQL 쿼리 수집
        sql_queries = []
        for h in validated:
            data = h.validation_data or {}
            sql_query = data.get("sql_query", "")
            if sql_query:
                sql_queries.append({
                    "hypothesis_id": h.id,
                    "factor": h.factor,
                    "sql": sql_query
                })
        progress.progress(50)

        # Step 4: 이벤트 매칭
        status.text("🎯 Event Matching (Scoring Algorithm)...")
        try:
            matched_events = analysis_agent.event_matcher.match(
                hypotheses=validated,
                region=region,
                min_score=0.3,
                top_k=5
            )
        except Exception as e:
            matched_events = {}
            st.warning(f"이벤트 매칭 오류: {e}")
        progress.progress(70)

        # Step 5: 추론 기반 답변 생성
        status.text("🧠 추론 기반 답변 생성 중...")
        details = analysis_agent._build_details(validated, matched_events, sql_queries)
        summary_result = analysis_agent._generate_summary(query, details, kpi_change)
        progress.progress(90)

        # ========== 2. 결과 표시 (분석 과정 먼저, 답변 나중에) ==========
        status.text("✅ 분석 완료!")
        progress.progress(100)

        # 🔍 분석 과정 먼저 (각 Step은 닫혀있음)
        st.markdown("### 🔍 분석 과정")

        # Step 1: KPI 변동 (닫혀있음)
        with st.expander("📊 Step 1: KPI 변동 현황", expanded=False):
            if kpi_change:
                change_direction = "증가 📈" if kpi_change.change_percent > 0 else "감소 📉"
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("이전 기간", f"{kpi_change.previous_value:,.0f}")
                with col2:
                    st.metric("현재 기간", f"{kpi_change.current_value:,.0f}")
                with col3:
                    st.metric("변화율", f"{kpi_change.change_percent:+.1f}%", delta=change_direction)
                st.caption(f"기간: {kpi_change.period_info}")
                st.markdown("**SQL Query:**")
                st.code(kpi_change.sql_query, language="sql")
            else:
                st.warning("KPI 변동 데이터를 계산할 수 없습니다.")

        # Step 2: 가설 생성 (닫혀있음) - 상세 정보 포함
        graph_based_count = sum(1 for h in hypotheses if h.graph_evidence.get("from_graph", False))
        with st.expander(f"💡 Step 2: 가설 생성 ({len(hypotheses)}개, Graph 기반: {graph_based_count}개)", expanded=False):
            for h in hypotheses:
                # 카테고리별 아이콘
                cat_icons = {"cost": "🔴", "revenue": "🟢", "pricing": "🔵", "external": "🟡"}
                cat_icon = cat_icons.get(h.category, "⚪")
                graph_icon = "🔗" if h.graph_evidence.get("from_graph", False) else "💭"

                st.markdown(f"#### {graph_icon} [{h.id}] {h.factor} {cat_icon}")

                # 인과관계 체인 (있으면)
                if hasattr(h, 'reasoning') and h.reasoning:
                    st.code(h.reasoning, language=None)

                # 상세 설명
                st.markdown(h.description)

                # Graph Evidence
                evidence = h.graph_evidence or {}
                if evidence.get("from_graph"):
                    mention_count = evidence.get("mention_count", 0)
                    relation_type = evidence.get("relation_type", "N/A")
                    relation_kr = "동비례" if relation_type == "PROPORTIONAL" else "역비례"
                    event_count = evidence.get("event_count", 0)
                    st.info(f"📊 **Graph Evidence**: 관계: {relation_kr} | 언급: {mention_count}회 | 이벤트: {event_count}개")

                # 관련 이벤트 목록
                if hasattr(h, 'related_events') and h.related_events:
                    st.markdown(f"**🔔 관련 이벤트 ({len(h.related_events)}건)**")
                    for event in h.related_events[:3]:
                        severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(event.severity, "⚪")
                        impact_text = "증가" if event.impact_direction == "INCREASES" else "감소"
                        regions = ", ".join([r for r in (event.target_regions or []) if r]) or "전체"
                        event_evidence = (event.evidence[:100] + "...") if event.evidence and len(event.evidence) > 100 else (event.evidence or "")

                        st.markdown(f"""
                        - {severity_emoji} **{event.name}** ({event.category})
                          - 영향: {h.factor} {impact_text} | 심각도: {event.severity} | 지역: {regions}
                          {f'- {event_evidence}' if event_evidence else ''}
                        """)

                st.markdown("---")

        # Step 3: 가설 검증 (닫혀있음) - SQL Query 포함
        validated_ids = {h.id for h in validated}
        rejected = [h for h in hypotheses if h.id not in validated_ids]

        with st.expander(f"✅ Step 3: 가설 검증 (검증: {len(validated)}개 / 기각: {len(rejected)}개)", expanded=False):
            # 검증된 가설
            if validated:
                st.markdown("##### ✅ 검증된 가설")
                for h in validated:
                    data = h.validation_data or {}
                    change = data.get("change_percent", 0)
                    prev_val = data.get('previous_value', 0)
                    curr_val = data.get('current_value', 0)
                    direction = data.get('direction', '')
                    sql_query = data.get('sql_query', '')

                    st.markdown(f"""
                    <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin-bottom: 8px; border-left: 4px solid #28a745;">
                        <strong>[{h.id}] {h.factor}</strong>: <span style="color: #155724;">{change:+.1f}%</span><br>
                        <span style="font-size: 0.9em;">{prev_val:,.0f} → {curr_val:,.0f} ({direction})</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if sql_query:
                        st.code(sql_query, language="sql")

            # 기각된 가설
            if rejected:
                st.markdown("##### ❌ 기각된 가설")
                for h in rejected:
                    data = h.validation_data or {}
                    change = data.get("change_percent", 0)
                    direction = data.get("direction", "unknown")
                    sql_query = data.get('sql_query', '')

                    if data:
                        if abs(change) < 5.0:
                            reject_reason = f"변동률 미달 ({change:+.1f}% < ±5%)"
                        elif h.direction.lower() == "increase" and direction == "decreased":
                            reject_reason = f"방향 불일치 (예상: 증가, 실제: 감소)"
                        elif h.direction.lower() == "decrease" and direction == "increased":
                            reject_reason = f"방향 불일치 (예상: 감소, 실제: 증가)"
                        else:
                            reject_reason = f"기타 ({direction})"
                    else:
                        reject_reason = "데이터 없음"

                    st.markdown(f"""
                    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 8px; border-left: 4px solid #ffc107;">
                        <strong>[{h.id}] {h.factor}</strong><br>
                        <span style="color: #856404;">기각 사유: {reject_reason}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if sql_query:
                        st.code(sql_query, language="sql")

        # Step 4: 이벤트 매칭 (닫혀있음)
        total_events = sum(len(v) for v in matched_events.values())
        with st.expander(f"🎯 Step 4: 이벤트 매칭 ({total_events}개)", expanded=False):
            for h_id, events in matched_events.items():
                st.markdown(f"**[{h_id}]** - {len(events)}개 이벤트")
                for ev in events[:5]:
                    score = ev.total_score
                    score_color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"

                    sources_html = ""
                    if ev.sources:
                        for src in ev.sources[:2]:
                            url = src.get('url', src.get('link', ''))
                            title = src.get('title', 'Link')
                            sources_html += f"<a href='{url}' target='_blank'>{title}</a><br>"

                    st.markdown(f"""
                    <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 8px;">
                        {score_color} <strong>{ev.event_name}</strong> (Score: {score:.2f})<br>
                        <span style="font-size: 0.9em;">
                            카테고리: {ev.event_category} | 영향: {ev.impact_type} → {ev.matched_factor}<br>
                            심각도: {ev.severity} | 지역: {', '.join(ev.target_regions) if ev.target_regions else '전체'}
                        </span>
                        {f'<br><span style="font-size: 0.85em;">출처: {sources_html}</span>' if sources_html else ''}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # 📝 답변 (분석 과정 아래에 표시)
        st.markdown("## 📝 분석 결과")
        summary = summary_result.get("summary", "") if isinstance(summary_result, dict) else summary_result
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #A50034; line-height: 1.8;">
        {summary.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

        # 결과 저장
        st.session_state.current_result = {
            "query": query,
            "intent": intent_result,
            "kpi_change": {
                "kpi_name": kpi_change.kpi_name if kpi_change else None,
                "change_percent": kpi_change.change_percent if kpi_change else None,
            } if kpi_change else None,
            "hypotheses": len(hypotheses),
            "validated": len(validated),
            "matched_events": sum(len(v) for v in matched_events.values()),
            "summary": summary_result.get("summary", "") if isinstance(summary_result, dict) else summary_result,
            "sources": summary_result.get("sources", []) if isinstance(summary_result, dict) else []
        }

    else:
        # Descriptive: Search Agent 사용
        status.text("🔍 데이터 검색 중...")

        search_agent = SearchAgent()

        # Intent Classifier 결과 사용 (중복 로직 제거)
        sub_intent = intent_result.get("sub_intent", "internal_data")
        is_event_query = intent_result.get("is_event_query", False)

        st.markdown("### 🔍 Step 2: Data Search")

        if is_event_query:
            st.info("🔎 Vector Search로 유사 이벤트 검색 중...")
            source = "vector"
        elif sub_intent == "external_data":
            st.info("📊 Knowledge Graph에서 검색 중...")
            source = "graph"
        else:
            st.info("📊 ERP 데이터베이스에서 검색 중...")
            source = "sql"

        progress.progress(40)

        context = AgentContext(
            query=query,
            metadata={"source": source, "top_k": 5}
        )

        result = search_agent.run(context)
        progress.progress(70)

        # 쿼리 표시
        if source != "vector":
            st.markdown("#### 생성된 쿼리")
            query_used = result.get("query", "")
            if source == "sql":
                st.code(query_used, language="sql")
            else:
                st.code(query_used, language="cypher")
        else:
            st.markdown("#### Vector Search 쿼리")
            st.code(f"의미적 유사도 검색: \"{query}\"", language="text")

        # 결과 표시
        st.markdown("#### 검색 결과")

        if result.get("success") and result.get("data"):
            data = result["data"]

            if source == "vector":
                # 벡터 검색 결과 표시 (이벤트 카드 형태)
                display_vector_search_results(data)
            elif isinstance(data, list) and data:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            else:
                st.json(data)
        else:
            st.error(f"검색 실패: {result.get('error', '알 수 없는 오류')}")

        progress.progress(100)

        st.session_state.current_result = {
            "query": query,
            "intent": intent_result,
            "data": result.get("data"),
            "source": source,
            "sql": result.get("query") if source == "sql" else None
        }

    status.text("✅ 분석 완료!")

    # 히스토리에 추가
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "mode": analysis_mode
    })


def main():
    """메인 함수"""
    init_session_state()

    # 헤더
    st.markdown('<p class="main-header">📊 LG HE Business Intelligence</p>', unsafe_allow_html=True)
    st.markdown("Multi-Agent System for Data Q&A and Report Generation")

    st.markdown("---")

    # 사이드바
    with st.sidebar:
        st.markdown("## ⚙️ 설정")

        # 시스템 상태
        st.markdown("### 시스템 상태")
        st.success("✅ Orchestrator 준비됨")
        st.success("✅ SQL Tool 준비됨")

        # Neo4j 연결 확인
        try:
            from agents.tools import GraphExecutor
            graph = GraphExecutor()
            result = graph.execute("RETURN 1 as test")
            if result.success:
                st.success("✅ Neo4j 연결됨")
            else:
                st.warning("⚠️ Neo4j 연결 실패")
        except:
            st.warning("⚠️ Neo4j 연결 실패")

        if INTENT_CLASSIFIER_AVAILABLE:
            st.success("✅ Intent Classifier 준비됨")
        else:
            st.info("ℹ️ 기본 Intent 분류 사용")

        st.markdown("---")

        # 예시 질문
        st.markdown("### 💡 예시 질문")

        example_queries = [
            "2024년 4분기 북미 영업이익이 왜 감소했어?",
            "2025년 Q3 매출 변동 원인 분석해줘",
            "2024년 4분기 총 매출은 얼마야?",
            "유럽 지역 원가 현황 알려줘",
            "최근 물류 관련 이벤트 알려줘",
            "관세 정책 관련 이슈가 뭐가 있어?",
        ]

        for eq in example_queries:
            if st.button(eq, key=f"example_{eq[:20]}"):
                st.session_state.example_query = eq

        st.markdown("---")

        # 히스토리
        st.markdown("### 📜 질문 히스토리")
        for item in st.session_state.history[-5:]:
            st.caption(f"[{item['timestamp']}] {item['query'][:30]}...")

    # 메인 영역
    col1, col2 = st.columns([4, 1])

    with col1:
        # 예시 질문이 선택되었으면 적용
        default_query = st.session_state.get("example_query", "")

        query = st.text_input(
            "질문을 입력하세요",
            value=default_query,
            placeholder="예: 2024년 4분기 북미 영업이익이 왜 감소했어?",
            key="query_input"
        )

        # 예시 질문 상태 초기화
        if "example_query" in st.session_state:
            del st.session_state.example_query

    with col2:
        analyze_button = st.button("🔍 분석", type="primary", use_container_width=True)

    st.markdown("---")

    # 분석 실행
    if analyze_button and query:
        with st.container():
            run_analysis(query)

    elif not query and analyze_button:
        st.warning("질문을 입력해주세요.")

    # 푸터
    st.markdown("---")
    st.caption("LG Electronics HE Business Intelligence System | Multi-Agent Architecture")


if __name__ == "__main__":
    main()
