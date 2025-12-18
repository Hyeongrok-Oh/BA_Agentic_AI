"""Factor 추출 - LLM 기반 Factor-Anchor 관계 추출"""

import json
import os
from typing import List, Optional
from dataclasses import dataclass

from .models import (
    FactorMention, SourceReference, RelationType,
    FactorCategory, Layer2Graph
)
from .pdf_extractor import Paragraph


# Factor-Anchor 관계 추출 프롬프트
EXTRACTION_PROMPT = """다음 문단에서 LG전자 TV(HE) 사업의 실적 변동 요인(Factor)과 KPI(Anchor) 간의 관계를 추출하세요.

**Anchor (KPI):**
- revenue: 매출, 매출액, 탑라인, 외형
- quantity: 판매수량, 출하량
- cost: 원가, 비용, 물류비

**추출 규칙:**
1. Factor는 실적에 영향을 주는 외부/내부 요인 (예: 환율, 수요, 경쟁, 패널가격 등)
2. 관계 타입:
   - PROPORTIONAL: Factor↑ → Anchor↑ (정비례)
   - INVERSELY_PROPORTIONAL: Factor↑ → Anchor↓ (반비례)
3. 문단에서 명시적으로 언급된 관계만 추출

**문단:**
{paragraph}

**응답 형식 (JSON):**
```json
{{
  "factors": [
    {{
      "factor_name": "요인명",
      "anchor_id": "revenue|quantity|cost",
      "relation_type": "PROPORTIONAL|INVERSELY_PROPORTIONAL",
      "evidence": "근거 문장"
    }}
  ]
}}
```

관계가 없으면 빈 배열 반환: {{"factors": []}}
"""


@dataclass
class ExtractionResult:
    """추출 결과"""
    factor_name: str
    anchor_id: str
    relation_type: str
    evidence: str


class FactorExtractor:
    """LLM 기반 Factor 추출"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 필요")

    def extract_from_paragraph(self, paragraph: Paragraph) -> List[FactorMention]:
        """문단에서 Factor-Anchor 관계 추출"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지 필요: pip install openai")

        client = OpenAI(api_key=self.api_key)

        prompt = EXTRACTION_PROMPT.format(paragraph=paragraph.text)

        message = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # 응답 파싱
        response_text = message.choices[0].message.content
        results = self._parse_response(response_text)

        # FactorMention으로 변환
        mentions = []
        for result in results:
            try:
                mention = FactorMention(
                    factor_name=result.factor_name,
                    anchor_id=result.anchor_id,
                    relation_type=RelationType(result.relation_type),
                    source=SourceReference(
                        doc_name=paragraph.doc_name,
                        doc_date=paragraph.doc_date,
                        doc_type=paragraph.doc_type,
                        paragraph=paragraph.text,
                        page_num=paragraph.page_num,
                    ),
                )
                mentions.append(mention)
            except (KeyError, ValueError) as e:
                print(f"파싱 오류: {e}")
                continue

        return mentions

    def _parse_response(self, response: str) -> List[ExtractionResult]:
        """LLM 응답 파싱"""
        # JSON 블록 추출
        import re
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            data = json.loads(json_str)
            factors = data.get("factors", [])
            return [
                ExtractionResult(
                    factor_name=f["factor_name"],
                    anchor_id=f["anchor_id"],
                    relation_type=f["relation_type"],
                    evidence=f.get("evidence", ""),
                )
                for f in factors
            ]
        except json.JSONDecodeError:
            return []


class Layer2Builder:
    """Layer 2 그래프 빌더"""

    def __init__(self, layer2_dir: str):
        from pathlib import Path
        self.layer2_dir = Path(layer2_dir)
        self.extractor = FactorExtractor()
        self.graph = Layer2Graph()

    def build(self, max_docs: Optional[int] = None) -> Layer2Graph:
        """전체 문서에서 Factor 추출"""
        from .pdf_extractor import DocumentProcessor

        processor = DocumentProcessor(self.layer2_dir)
        doc_count = 0
        para_count = 0

        print("TV 관련 문단에서 Factor 추출 시작...")

        for paragraph in processor.process_all_documents():
            para_count += 1

            # 진행 상황 출력
            if para_count % 10 == 0:
                print(f"  처리 중: {para_count}개 문단...")

            try:
                mentions = self.extractor.extract_from_paragraph(paragraph)
                for mention in mentions:
                    self.graph.add_mention(mention)
            except Exception as e:
                print(f"  추출 오류: {e}")

            # 문서 수 제한
            if max_docs and doc_count >= max_docs:
                break

        # 관계 집계
        self.graph.aggregate_relations()

        print(f"\n완료: {self.graph.summary()}")
        return self.graph

    def build_from_paragraphs(self, paragraphs: List[Paragraph]) -> Layer2Graph:
        """주어진 문단들에서 Factor 추출"""
        for para in paragraphs:
            try:
                mentions = self.extractor.extract_from_paragraph(para)
                for mention in mentions:
                    self.graph.add_mention(mention)
            except Exception as e:
                print(f"추출 오류: {e}")

        self.graph.aggregate_relations()
        return self.graph
