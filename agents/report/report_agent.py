"""
Report Agent - 보고서 생성 에이전트

역할:
- Analysis Agent를 다중 호출하여 섹션별 분석 수행
- 보고서 템플릿에 맞게 결과 조합
- 최종 보고서 포맷팅
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..base import BaseAgent, AgentContext
from ..search_agent import SearchAgent
from ..analysis import AnalysisAgent


@dataclass
class ReportSection:
    """보고서 섹션"""
    title: str
    query: str
    section_type: str = "analysis"  # analysis, search, custom
    content: str = ""
    data: Dict = field(default_factory=dict)


@dataclass
class Report:
    """보고서"""
    title: str
    period: str
    generated_at: str
    sections: List[ReportSection]
    summary: str = ""


# 보고서 템플릿
REPORT_TEMPLATES = {
    "quarterly_performance": {
        "title": "{period} 실적 분석 보고서",
        "sections": [
            {"title": "1. 매출 현황", "query": "매출 현황 및 전년 대비 변화", "type": "search"},
            {"title": "2. 원가 분석", "query": "원가 구조 및 변동 분석", "type": "search"},
            {"title": "3. 수익성 분석", "query": "영업이익 변동 원인 분석", "type": "analysis"},
            {"title": "4. 시사점", "query": "분석 결과 종합 및 시사점", "type": "summary"},
        ]
    },
    "profit_decline_analysis": {
        "title": "{period} 영업이익 하락 원인 분석",
        "sections": [
            {"title": "1. 개요", "query": "영업이익 현황 요약", "type": "search"},
            {"title": "2. 원가 측면", "query": "원가 증가 원인 분석", "type": "analysis"},
            {"title": "3. 매출 측면", "query": "매출 변동 원인 분석", "type": "analysis"},
            {"title": "4. 외부 요인", "query": "외부 환경 영향 분석", "type": "analysis"},
            {"title": "5. 결론", "query": "종합 결론 및 대응 방안", "type": "summary"},
        ]
    },
    "regional_analysis": {
        "title": "{region} 지역 {period} 성과 분석",
        "sections": [
            {"title": "1. 지역 매출 현황", "query": "{region} 매출 및 판매량 현황", "type": "search"},
            {"title": "2. 수익성 분석", "query": "{region} 영업이익 변동 원인", "type": "analysis"},
            {"title": "3. 경쟁 환경", "query": "{region} 시장 경쟁 현황", "type": "analysis"},
            {"title": "4. 전략 제언", "query": "{region} 향후 전략 방향", "type": "summary"},
        ]
    }
}

SECTION_SUMMARY_PROMPT = """다음 섹션들의 분석 결과를 종합하여 최종 요약을 작성하세요.

## 보고서 주제
{title}

## 섹션별 분석 결과
{sections_content}

## 작성 지침
1. 핵심 발견사항 3-5가지로 요약
2. 각 발견사항에 대한 근거 수치 포함
3. 향후 대응 방향 제시
4. 한국어로 3-4문단 분량

## 종합 요약
"""


class ReportAgent(BaseAgent):
    """보고서 생성 에이전트"""

    name = "report_agent"
    description = "Analysis Agent를 다중 호출하여 종합 보고서를 생성합니다."

    def __init__(self, api_key: str = None, db_path: str = None):
        super().__init__(api_key)

        # 하위 에이전트 초기화
        self.search_agent = SearchAgent(api_key, db_path)
        self.analysis_agent = AnalysisAgent(api_key, db_path)

        self.add_sub_agent(self.search_agent)
        self.add_sub_agent(self.analysis_agent)

    def generate(
        self,
        template: str,
        period: Dict = None,
        region: str = None,
        company: str = "LGE",
        custom_sections: List[Dict] = None,
        verbose: bool = True
    ) -> Report:
        """
        보고서 생성

        Args:
            template: 템플릿 이름 (quarterly_performance, profit_decline_analysis 등)
            period: {"year": 2024, "quarter": 4}
            region: 지역 코드
            company: 회사 코드
            custom_sections: 커스텀 섹션 리스트
            verbose: 상세 출력

        Returns:
            Report 객체
        """
        if not period:
            period = {"year": 2024, "quarter": 4}

        period_str = f"{period['year']}년 Q{period['quarter']}"

        # 템플릿 가져오기
        if custom_sections:
            template_data = {
                "title": f"{period_str} 분석 보고서",
                "sections": custom_sections
            }
        else:
            template_data = REPORT_TEMPLATES.get(template)
            if not template_data:
                raise ValueError(f"Unknown template: {template}")

        # 제목 포맷팅
        title = template_data["title"].format(
            period=period_str,
            region=region or "전체"
        )

        if verbose:
            print("=" * 60)
            print(f"보고서 생성: {title}")
            print("=" * 60)

        # 섹션별 분석 실행
        sections = []
        for i, section_def in enumerate(template_data["sections"], 1):
            section_title = section_def["title"].format(
                period=period_str,
                region=region or "전체"
            )
            section_query = section_def["query"].format(
                period=period_str,
                region=region or "전체"
            )
            section_type = section_def.get("type", "analysis")

            if verbose:
                print(f"\n[섹션 {i}] {section_title}")
                print(f"  쿼리: {section_query}")

            section = ReportSection(
                title=section_title,
                query=section_query,
                section_type=section_type
            )

            # 섹션 타입별 처리
            if section_type == "search":
                result = self._process_search_section(
                    section_query, period, region, verbose
                )
                section.content = result.get("content", "")
                section.data = result.get("data", {})

            elif section_type == "analysis":
                result = self._process_analysis_section(
                    section_query, period, region, company, verbose
                )
                section.content = result.get("summary", "")
                section.data = result.get("details", {})

            elif section_type == "summary":
                # summary는 나중에 처리
                pass

            sections.append(section)

            if verbose and section.content:
                print(f"  결과: {section.content[:100]}...")

        # 최종 요약 생성 (summary 섹션)
        for section in sections:
            if section.section_type == "summary":
                section.content = self._generate_report_summary(
                    title, sections, verbose
                )

        # Report 객체 생성
        report = Report(
            title=title,
            period=period_str,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sections=sections
        )

        # 전체 요약
        report.summary = self._generate_report_summary(title, sections, verbose)

        if verbose:
            print("\n" + "=" * 60)
            print("보고서 생성 완료!")
            print("=" * 60)

        return report

    def _process_search_section(
        self,
        query: str,
        period: Dict,
        region: str,
        verbose: bool
    ) -> Dict:
        """검색 섹션 처리"""
        context = AgentContext(
            query=query,
            metadata={"source": "sql"}
        )

        result = self.search_agent.run(context)

        content = ""
        if result.get("success") and result.get("data"):
            data = result["data"]
            if isinstance(data, list) and data:
                # 간단한 요약 생성
                content = self._summarize_search_result(query, data)

        return {
            "content": content,
            "data": result.get("data", {})
        }

    def _process_analysis_section(
        self,
        query: str,
        period: Dict,
        region: str,
        company: str,
        verbose: bool
    ) -> Dict:
        """분석 섹션 처리"""
        result = self.analysis_agent.analyze(
            question=query,
            period=period,
            region=region,
            company=company,
            verbose=False  # 보고서 모드에서는 개별 분석 출력 끔
        )

        return {
            "summary": result.summary,
            "details": result.details,
            "validated_count": len(result.validated_hypotheses)
        }

    def _summarize_search_result(self, query: str, data: List[Dict]) -> str:
        """검색 결과 요약"""
        if not data:
            return "데이터가 없습니다."

        # 데이터 요약 프롬프트
        data_str = str(data[:10])  # 상위 10개만

        prompt = f"""다음 검색 결과를 한 문단으로 요약하세요.

질문: {query}
데이터: {data_str}

요약:"""

        return self._call_llm(
            prompt=prompt,
            system_prompt="데이터 분석 결과를 간결하게 요약하세요.",
            model="gpt-4o-mini",
            max_tokens=300
        )

    def _generate_report_summary(
        self,
        title: str,
        sections: List[ReportSection],
        verbose: bool
    ) -> str:
        """보고서 종합 요약 생성"""
        sections_content = ""
        for section in sections:
            if section.content and section.section_type != "summary":
                sections_content += f"\n### {section.title}\n{section.content}\n"

        if not sections_content:
            return "분석 결과가 없습니다."

        prompt = SECTION_SUMMARY_PROMPT.format(
            title=title,
            sections_content=sections_content
        )

        return self._call_llm(
            prompt=prompt,
            system_prompt="당신은 LG전자 HE사업부의 전략 분석가입니다.",
            model="gpt-4o",
            max_tokens=1500
        )

    def format_markdown(self, report: Report) -> str:
        """보고서를 Markdown 형식으로 포맷팅"""
        md = f"# {report.title}\n\n"
        md += f"**생성일시**: {report.generated_at}\n"
        md += f"**분석 기간**: {report.period}\n\n"
        md += "---\n\n"

        for section in report.sections:
            md += f"## {section.title}\n\n"
            md += f"{section.content}\n\n"

        if report.summary:
            md += "---\n\n"
            md += "## 종합 요약\n\n"
            md += f"{report.summary}\n"

        return md

    def run(self, context: AgentContext) -> Dict[str, Any]:
        """Agent 실행"""
        metadata = context.metadata or {}

        report = self.generate(
            template=metadata.get("template", "quarterly_performance"),
            period=metadata.get("period", {"year": 2024, "quarter": 4}),
            region=metadata.get("region"),
            company=metadata.get("company", "LGE"),
            custom_sections=metadata.get("custom_sections"),
            verbose=metadata.get("verbose", True)
        )

        return {
            "title": report.title,
            "period": report.period,
            "sections_count": len(report.sections),
            "summary": report.summary,
            "markdown": self.format_markdown(report)
        }
