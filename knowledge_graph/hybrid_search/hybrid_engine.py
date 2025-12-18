"""
Hybrid Search Engine - 가설 생성 → 검증 → 그래프 탐색 통합
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from openai import OpenAI

from .hypothesis_generator import HypothesisGenerator, Hypothesis
from .hypothesis_validator import HypothesisValidator
from .graph_searcher import GraphSearcher, GraphEvidence


@dataclass
class AnalysisResult:
    """분석 결과"""
    question: str
    hypotheses: List[Hypothesis]
    validated_hypotheses: List[Hypothesis]
    graph_evidences: Dict[str, List[GraphEvidence]]
    summary: str = ""
    details: List[Dict] = field(default_factory=list)


class HybridSearchEngine:
    """가설 기반 하이브리드 검색 엔진"""

    def __init__(
        self,
        db_path: str = "/Users/hyeongrokoh/BI/sql/lge_he_erp.db",
        api_key: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.db_path = db_path

        # 컴포넌트 초기화
        self.hypothesis_generator = HypothesisGenerator(self.api_key)
        self.hypothesis_validator = HypothesisValidator(db_path, self.api_key)
        self.graph_searcher = GraphSearcher()
        self.llm_client = OpenAI(api_key=self.api_key)

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

        # Step 1: 가설 생성
        if verbose:
            print("\n📊 Step 1: 가설 생성 중...")

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

        # Step 2: 가설 검증
        if verbose:
            print("\n🔍 Step 2: 가설 검증 중 (SQL Agent)...")

        validated = self.hypothesis_validator.validate_hypotheses(
            hypotheses=hypotheses,
            period=period,
            region=region,
            threshold=5.0
        )

        if verbose:
            print(f"  검증된 가설: {len(validated)}개")
            for h in validated:
                data = h.validation_data or {}
                print(f"    - [{h.id}] {h.factor}: {data.get('details', '')}")

        # Step 3: 그래프 검색
        if verbose:
            print("\n🔗 Step 3: Graph 검색 중 (Neo4j)...")

        graph_evidences = {}
        try:
            graph_evidences = self.graph_searcher.search_for_hypotheses(
                hypotheses=validated,
                region=region
            )

            if verbose:
                for h_id, evidences in graph_evidences.items():
                    print(f"  [{h_id}] 관련 이벤트: {len(evidences)}개")
                    for ev in evidences[:3]:
                        print(f"    - {ev.event_name} ({ev.event_category})")

        except Exception as e:
            if verbose:
                print(f"  ⚠️ Graph 검색 오류: {e}")

        # Step 4: 결과 종합
        if verbose:
            print("\n📝 Step 4: 결과 종합 중...")

        result = AnalysisResult(
            question=question,
            hypotheses=hypotheses,
            validated_hypotheses=validated,
            graph_evidences=graph_evidences
        )

        # 상세 분석 결과 구성
        result.details = self._build_details(validated, graph_evidences)

        # LLM으로 요약 생성
        result.summary = self._generate_summary(question, result.details)

        if verbose:
            print("\n" + "=" * 60)
            print("분석 완료!")
            print("=" * 60)

        return result

    def _build_details(
        self,
        validated: List[Hypothesis],
        graph_evidences: Dict[str, List[GraphEvidence]]
    ) -> List[Dict]:
        """상세 분석 결과 구성"""

        details = []

        for hypothesis in validated:
            h_data = hypothesis.validation_data or {}

            detail = {
                "factor": hypothesis.factor,
                "category": hypothesis.category,
                "description": hypothesis.description,
                "change_percent": h_data.get("change_percent", 0),
                "previous_value": h_data.get("previous_value", 0),
                "current_value": h_data.get("current_value", 0),
                "direction": h_data.get("direction", ""),
                "related_events": []
            }

            # 관련 이벤트 추가
            evidences = graph_evidences.get(hypothesis.id, [])
            for ev in evidences[:5]:
                detail["related_events"].append({
                    "name": ev.event_name,
                    "category": ev.event_category,
                    "severity": ev.event_severity,
                    "impact": ev.impact_type,
                    "evidence": ev.evidence[:200] if ev.evidence else ""
                })

            details.append(detail)

        # 변화율 기준 정렬
        details.sort(key=lambda x: abs(x["change_percent"]), reverse=True)

        return details

    def _generate_summary(self, question: str, details: List[Dict]) -> str:
        """LLM으로 분석 요약 생성"""

        if not details:
            return "검증된 원인을 찾지 못했습니다."

        # 프롬프트 구성
        details_text = ""
        for i, d in enumerate(details[:5], 1):
            details_text += f"""
{i}. **{d['factor']}** ({d['category']})
   - 변화: {d['change_percent']:+.1f}%
   - 이전: {d['previous_value']:,.0f} → 현재: {d['current_value']:,.0f}
"""
            if d['related_events']:
                details_text += "   - 관련 이벤트:\n"
                for ev in d['related_events'][:2]:
                    details_text += f"     * {ev['name']} ({ev['category']})\n"

        prompt = f"""다음 분석 결과를 바탕으로 사용자 질문에 대한 답변을 작성하세요.

## 질문
{question}

## 분석 결과
{details_text}

## 작성 지침
1. 핵심 원인을 변화율이 큰 순서대로 설명
2. 각 원인에 대한 외부 이벤트/요인 연결
3. 구체적인 수치 포함
4. 한국어로 2-3문단 분량

## 답변
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 LG전자 HE사업부의 재무 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"요약 생성 오류: {e}"

    def analyze_from_intent(self, intent_result: Dict, verbose: bool = True) -> AnalysisResult:
        """Intent Classifier 결과로부터 분석 실행"""

        entities = intent_result.get("extracted_entities", {}) or {}

        # 기간 추출
        period = entities.get("period")
        if period:
            period_dict = {
                "year": period.get("year", 2024),
                "quarter": period.get("quarter", 4)
            }
        else:
            period_dict = {"year": 2024, "quarter": 4}

        # 지역 추출
        region = entities.get("region")
        if isinstance(region, list):
            region = region[0] if region else None

        # 회사 추출
        company = entities.get("company", "LGE")

        # 질문 재구성
        thinking = intent_result.get("thinking", "")
        question = thinking if thinking else "KPI 변동 원인 분석"

        return self.analyze(
            question=question,
            period=period_dict,
            region=region,
            company=company,
            verbose=verbose
        )


def run_analysis(question: str, year: int = 2024, quarter: int = 4, region: str = None):
    """간편 분석 실행 함수"""

    engine = HybridSearchEngine()

    result = engine.analyze(
        question=question,
        period={"year": year, "quarter": quarter},
        region=region,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("📋 분석 요약")
    print("=" * 60)
    print(result.summary)

    return result
