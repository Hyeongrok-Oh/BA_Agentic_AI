"""Factor-Factor 관계 추출 - LLM 기반"""

import json
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .pdf_extractor import Paragraph, DocumentProcessor


# Factor-Factor 관계 추출 프롬프트
FACTOR_RELATION_PROMPT = """다음 문단에서 LG전자 TV(HE) 사업의 실적 변동 요인(Factor)들 간의 인과관계를 추출하세요.

**추출 규칙:**
1. Factor는 실적에 영향을 주는 외부/내부 요인 (예: 환율, 수요, 경쟁, 패널가격, 물류비 등)
2. Factor 간의 인과관계를 추출:
   - CAUSES: A가 B를 유발 (A → B)
   - AMPLIFIES: A가 B를 강화/증폭
   - MITIGATES: A가 B를 완화/감소
3. 문단에서 명시적 또는 암시적으로 언급된 관계만 추출
4. Factor명은 간결하게 (예: "환율", "경쟁심화", "패널가격", "물류비")

**문단:**
{paragraph}

**응답 형식 (JSON):**
```json
{{
  "relations": [
    {{
      "source_factor": "원인 Factor명",
      "target_factor": "결과 Factor명",
      "relation_type": "CAUSES|AMPLIFIES|MITIGATES",
      "evidence": "근거 문장"
    }}
  ]
}}
```

관계가 없으면 빈 배열 반환: {{"relations": []}}
"""


@dataclass
class FactorRelation:
    """Factor-Factor 관계"""
    source_factor: str
    target_factor: str
    relation_type: str
    evidence: str
    doc_name: str
    doc_date: str


@dataclass
class FactorRelationGraph:
    """Factor-Factor 관계 그래프"""
    relations: List[FactorRelation] = field(default_factory=list)

    def add_relation(self, relation: FactorRelation):
        self.relations.append(relation)

    def aggregate(self) -> List[dict]:
        """관계 집계"""
        relation_map: Dict[Tuple[str, str, str], dict] = {}

        for rel in self.relations:
            key = (rel.source_factor, rel.target_factor, rel.relation_type)
            if key not in relation_map:
                relation_map[key] = {
                    "source_factor": rel.source_factor,
                    "target_factor": rel.target_factor,
                    "relation_type": rel.relation_type,
                    "mention_count": 0,
                    "evidences": [],
                    "sources": []
                }
            relation_map[key]["mention_count"] += 1
            relation_map[key]["evidences"].append(rel.evidence)
            if rel.doc_name not in relation_map[key]["sources"]:
                relation_map[key]["sources"].append(rel.doc_name)

        return sorted(
            relation_map.values(),
            key=lambda x: x["mention_count"],
            reverse=True
        )

    def summary(self) -> dict:
        return {
            "total_relations": len(self.relations),
            "unique_relations": len(set(
                (r.source_factor, r.target_factor, r.relation_type)
                for r in self.relations
            ))
        }


class FactorRelationExtractor:
    """Factor-Factor 관계 추출기"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 필요")

    def extract_from_paragraph(self, paragraph: Paragraph) -> List[FactorRelation]:
        """문단에서 Factor-Factor 관계 추출"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지 필요: pip install openai")

        client = OpenAI(api_key=self.api_key)
        prompt = FACTOR_RELATION_PROMPT.format(paragraph=paragraph.text)

        message = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.choices[0].message.content
        return self._parse_response(response_text, paragraph)

    def _parse_response(self, response: str, paragraph: Paragraph) -> List[FactorRelation]:
        """LLM 응답 파싱"""
        import re

        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            data = json.loads(json_str)
            relations = []
            for r in data.get("relations", []):
                relations.append(FactorRelation(
                    source_factor=r["source_factor"],
                    target_factor=r["target_factor"],
                    relation_type=r["relation_type"],
                    evidence=r.get("evidence", ""),
                    doc_name=paragraph.doc_name,
                    doc_date=str(paragraph.doc_date),
                ))
            return relations
        except json.JSONDecodeError:
            return []


class FactorRelationBuilder:
    """Factor-Factor 관계 빌드"""

    def __init__(self, layer2_dir: str):
        self.layer2_dir = Path(layer2_dir)
        self.extractor = FactorRelationExtractor()
        self.graph = FactorRelationGraph()

    def build(self, max_paragraphs: Optional[int] = None) -> FactorRelationGraph:
        """전체 문서에서 Factor-Factor 관계 추출"""
        processor = DocumentProcessor(self.layer2_dir)
        para_count = 0

        print("TV 관련 문단에서 Factor-Factor 관계 추출 시작...")

        for paragraph in processor.process_all_documents():
            para_count += 1

            if para_count % 10 == 0:
                print(f"  처리 중: {para_count}개 문단...")

            try:
                relations = self.extractor.extract_from_paragraph(paragraph)
                for relation in relations:
                    self.graph.add_relation(relation)
            except Exception as e:
                print(f"  추출 오류: {e}")

            if max_paragraphs and para_count >= max_paragraphs:
                break

        print(f"\n완료: {self.graph.summary()}")
        return self.graph


def normalize_factor_relations(
    relations: List[dict],
    normalization_path: Path
) -> List[dict]:
    """Factor-Factor 관계 정규화"""
    # 정규화 사전 로드
    with open(normalization_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    # 역매핑 생성
    reverse_mapping = {}
    for group_name, group_data in mapping.items():
        if group_name == "exclude":
            continue
        canonical = group_data.get("canonical", group_name)
        for variant in group_data.get("variants", []):
            reverse_mapping[variant.lower()] = canonical

    exclude_list = [e.lower() for e in mapping.get("exclude", [])]

    def normalize_name(name: str) -> Optional[str]:
        name_lower = name.lower()
        if name_lower in exclude_list:
            return None
        return reverse_mapping.get(name_lower, name)

    # 정규화 적용
    normalized = []
    relation_map = {}

    for rel in relations:
        source = normalize_name(rel["source_factor"])
        target = normalize_name(rel["target_factor"])

        if not source or not target:
            continue
        if source == target:
            continue

        key = (source, target, rel["relation_type"])
        if key not in relation_map:
            relation_map[key] = {
                "source_factor": source,
                "target_factor": target,
                "relation_type": rel["relation_type"],
                "mention_count": 0,
                "sources": []
            }

        relation_map[key]["mention_count"] += rel.get("mention_count", 1)
        relation_map[key]["sources"].extend(rel.get("sources", []))

    # 중복 source 제거 및 정렬
    for rel in relation_map.values():
        rel["sources"] = list(set(rel["sources"]))
        normalized.append(rel)

    return sorted(normalized, key=lambda x: x["mention_count"], reverse=True)


def build_factor_relations(
    max_paragraphs: Optional[int] = None,
    output_file: str = "factor_relations.json"
):
    """Factor-Factor 관계 빌드 및 저장"""
    layer2_dir = Path(__file__).parent

    builder = FactorRelationBuilder(str(layer2_dir))
    graph = builder.build(max_paragraphs=max_paragraphs)

    # 집계
    aggregated = graph.aggregate()

    # 정규화
    normalization_path = layer2_dir / "factor_normalization.yaml"
    if normalization_path.exists():
        print("\nFactor 정규화 적용 중...")
        normalized = normalize_factor_relations(aggregated, normalization_path)
    else:
        normalized = aggregated

    # 결과 저장
    result = {
        "summary": {
            "total_extracted": len(graph.relations),
            "aggregated": len(aggregated),
            "normalized": len(normalized),
        },
        "relations": normalized,
    }

    output_path = layer2_dir / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== Factor-Factor 관계 추출 완료 ===")
    print(f"추출된 관계: {result['summary']['total_extracted']}개")
    print(f"집계 후: {result['summary']['aggregated']}개")
    print(f"정규화 후: {result['summary']['normalized']}개")
    print(f"결과 저장: {output_path}")

    print("\n=== 상위 관계 ===")
    for rel in normalized[:15]:
        print(f"  {rel['source_factor']} --{rel['relation_type']}--> {rel['target_factor']} ({rel['mention_count']}회)")

    return result


if __name__ == "__main__":
    build_factor_relations()
