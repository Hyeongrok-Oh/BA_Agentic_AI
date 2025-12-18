"""Factor-Factor 관계 정규화 및 Neo4j 적재"""

import json
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from neo4j import GraphDatabase

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from knowledge_graph.config import BaseConfig


def load_normalization_mapping(normalization_path: Path) -> Tuple[Dict[str, str], List[str]]:
    """정규화 매핑 로드"""
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

    return reverse_mapping, exclude_list


def normalize_factor_name(name: str, reverse_mapping: Dict[str, str], exclude_list: List[str]) -> Optional[str]:
    """Factor 이름 정규화"""
    name_lower = name.lower()
    if name_lower in exclude_list:
        return None
    return reverse_mapping.get(name_lower, name)


def renormalize_factor_relations(
    input_path: Path,
    normalization_path: Path
) -> List[dict]:
    """Factor-Factor 관계 재정규화"""
    # 입력 로드
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 정규화 매핑 로드
    reverse_mapping, exclude_list = load_normalization_mapping(normalization_path)

    # 재정규화
    relation_map = {}

    for rel in data.get("relations", []):
        source = normalize_factor_name(rel["source_factor"], reverse_mapping, exclude_list)
        target = normalize_factor_name(rel["target_factor"], reverse_mapping, exclude_list)

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
    normalized = []
    for rel in relation_map.values():
        rel["sources"] = list(set(rel["sources"]))
        normalized.append(rel)

    return sorted(normalized, key=lambda x: x["mention_count"], reverse=True)


class FactorRelationNeo4jLoader:
    """Factor-Factor 관계 Neo4j 로더"""

    def __init__(self, config: Optional[BaseConfig] = None):
        self.config = config or BaseConfig()
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def clear_factor_relations(self):
        """기존 Factor-Factor 관계 삭제"""
        query = "MATCH (:Factor)-[r:INFLUENCES]->(:Factor) DELETE r"

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query)
            summary = result.consume()
            print(f"  삭제된 관계: {summary.counters.relationships_deleted}개")

    def ensure_factors_exist(self, relations: List[dict]) -> int:
        """관계에 등장하는 Factor 노드 확인/생성"""
        # 모든 Factor 이름 수집
        factors = set()
        for rel in relations:
            factors.add(rel["source_factor"])
            factors.add(rel["target_factor"])

        # 기존에 없는 Factor만 생성
        query = """
        UNWIND $factors AS factor_name
        MERGE (f:Factor {id: toLower(replace(factor_name, ' ', '_'))})
        ON CREATE SET f.name = factor_name,
                      f.category = 'extracted',
                      f.mention_count = 0
        RETURN count(f) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, factors=list(factors))
            record = result.single()
            return record["count"] if record else 0

    def load_relations(self, relations: List[dict]) -> int:
        """Factor-Factor 관계 생성"""
        query = """
        UNWIND $relations AS rel
        MATCH (source:Factor {id: toLower(replace(rel.source_factor, ' ', '_'))})
        MATCH (target:Factor {id: toLower(replace(rel.target_factor, ' ', '_'))})
        MERGE (source)-[r:INFLUENCES {type: rel.relation_type}]->(target)
        SET r.mention_count = rel.mention_count,
            r.source_count = size(rel.sources)
        RETURN count(r) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, relations=relations)
            record = result.single()
            return record["count"] if record else 0

    def verify_load(self) -> dict:
        """적재 결과 검증"""
        queries = {
            "relation_count": "MATCH (:Factor)-[r:INFLUENCES]->(:Factor) RETURN count(r) as count",
            "top_relations": """
                MATCH (s:Factor)-[r:INFLUENCES]->(t:Factor)
                RETURN s.name as source, r.type as type, t.name as target, r.mention_count as mentions
                ORDER BY r.mention_count DESC
                LIMIT 10
            """,
            "relation_types": """
                MATCH (:Factor)-[r:INFLUENCES]->(:Factor)
                RETURN r.type as type, count(r) as count
                ORDER BY count DESC
            """,
        }

        results = {}
        with self.driver.session(database=self.config.neo4j_database) as session:
            for key, query in queries.items():
                result = session.run(query)
                if key.endswith("_count"):
                    record = result.single()
                    results[key] = record["count"] if record else 0
                else:
                    results[key] = [dict(record) for record in result]

        return results


def load_factor_relations_to_neo4j(
    input_file: str = "factor_relations.json",
    output_file: str = "factor_relations_normalized.json"
):
    """Factor-Factor 관계 재정규화 및 Neo4j 적재"""
    layer2_dir = Path(__file__).parent
    input_path = layer2_dir / input_file
    normalization_path = layer2_dir / "factor_normalization.yaml"
    output_path = layer2_dir / output_file

    print("=== Factor-Factor 관계 정규화 및 Neo4j 적재 ===")

    # 재정규화
    print("\n1. 관계 재정규화...")
    normalized = renormalize_factor_relations(input_path, normalization_path)
    print(f"  정규화된 관계: {len(normalized)}개")

    # 결과 저장
    result = {
        "summary": {"normalized_relations": len(normalized)},
        "relations": normalized
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  저장: {output_path}")

    # Neo4j 적재
    print("\n2. Neo4j 적재...")
    with FactorRelationNeo4jLoader() as loader:
        # 기존 관계 삭제
        print("  기존 Factor-Factor 관계 삭제...")
        loader.clear_factor_relations()

        # Factor 노드 확인/생성
        print("  Factor 노드 확인/생성...")
        factor_count = loader.ensure_factors_exist(normalized)
        print(f"  처리된 Factor: {factor_count}개")

        # 관계 생성
        print("  INFLUENCES 관계 생성...")
        rel_count = loader.load_relations(normalized)
        print(f"  생성된 관계: {rel_count}개")

        # 검증
        print("\n3. 적재 결과 검증...")
        verification = loader.verify_load()

        print(f"\n=== 적재 완료 ===")
        print(f"INFLUENCES 관계: {verification['relation_count']}개")

        print("\n--- 관계 유형별 ---")
        for rt in verification.get("relation_types", []):
            print(f"  {rt['type']}: {rt['count']}개")

        print("\n--- 상위 관계 ---")
        for r in verification.get("top_relations", []):
            print(f"  {r['source']} --{r['type']}--> {r['target']}: {r['mentions']}회")

    return verification


if __name__ == "__main__":
    load_factor_relations_to_neo4j()
