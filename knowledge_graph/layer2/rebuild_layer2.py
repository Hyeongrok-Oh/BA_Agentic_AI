"""Layer 2 전체 재구축 - 정규화된 Factor와 관계 적재"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from neo4j import GraphDatabase

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from knowledge_graph.config import BaseConfig


def load_normalization_mapping(normalization_path: Path) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """정규화 매핑 로드: (reverse_mapping, category_mapping, exclude_list)"""
    with open(normalization_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    reverse_mapping = {}
    category_mapping = {}

    for group_name, group_data in mapping.items():
        if group_name == "exclude":
            continue
        canonical = group_data.get("canonical", group_name)
        category = group_data.get("category", "macro_economy")
        category_mapping[canonical] = category

        for variant in group_data.get("variants", []):
            reverse_mapping[variant.lower()] = canonical

    exclude_list = [e.lower() for e in mapping.get("exclude", [])]
    return reverse_mapping, category_mapping, exclude_list


def normalize_name(name: str, reverse_mapping: Dict[str, str], exclude_list: List[str]) -> Optional[str]:
    """Factor 이름 정규화"""
    name_lower = name.lower()
    if name_lower in exclude_list:
        return None
    return reverse_mapping.get(name_lower, name)


class Layer2Rebuilder:
    """Layer 2 전체 재구축"""

    def __init__(self, config: Optional[BaseConfig] = None):
        self.config = config or BaseConfig()
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        self.layer2_dir = Path(__file__).parent

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def clear_all_factors(self):
        """모든 Factor 노드와 관계 삭제"""
        queries = [
            "MATCH (:Factor)-[r:INFLUENCES]->(:Factor) DELETE r",
            "MATCH (:Factor)-[r:AFFECTS]->(:Anchor) DELETE r",
            "MATCH (f:Factor) DELETE f",
        ]

        with self.driver.session(database=self.config.neo4j_database) as session:
            for query in queries:
                result = session.run(query)
                summary = result.consume()
                deleted = summary.counters.relationships_deleted + summary.counters.nodes_deleted
                if deleted > 0:
                    print(f"  삭제: {deleted}개")

    def rebuild(self):
        """Layer 2 전체 재구축"""
        print("=== Layer 2 전체 재구축 ===\n")

        # 정규화 매핑 로드
        norm_path = self.layer2_dir / "factor_normalization_extended.yaml"
        reverse_mapping, category_mapping, exclude_list = load_normalization_mapping(norm_path)
        print(f"정규화 사전 로드: {len(reverse_mapping)}개 매핑")

        # 1. 기존 데이터 삭제
        print("\n1. 기존 Factor 데이터 삭제...")
        self.clear_all_factors()

        # 2. Factor-Anchor 관계 정규화 및 적재
        print("\n2. Factor-Anchor 관계 재정규화...")
        fa_result = self._rebuild_factor_anchor(reverse_mapping, category_mapping, exclude_list)

        # 3. Factor-Factor 관계 정규화 및 적재
        print("\n3. Factor-Factor 관계 재정규화...")
        ff_result = self._rebuild_factor_factor(reverse_mapping, category_mapping, exclude_list)

        # 4. 검증
        print("\n4. 적재 결과 검증...")
        verification = self._verify()

        return {
            "factor_anchor": fa_result,
            "factor_factor": ff_result,
            "verification": verification
        }

    def _rebuild_factor_anchor(self, reverse_mapping, category_mapping, exclude_list) -> dict:
        """Factor-Anchor 관계 재구축"""
        input_path = self.layer2_dir / "layer2_normalized.json"

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", [])
        relations = data.get("relations", [])

        # Factor 노드 생성
        factor_nodes = []
        for f in factors:
            name = normalize_name(f["name"], reverse_mapping, exclude_list)
            if not name:
                continue
            factor_nodes.append({
                "id": name.lower().replace(" ", "_"),
                "name": name,
                "category": category_mapping.get(name, f.get("category", "macro_economy")),
                "mention_count": f.get("mention_count", 1),
                "sources": f.get("sources", [])
            })

        # Factor 노드 적재 (중복 병합)
        factor_map = {}
        for f in factor_nodes:
            fid = f["id"]
            if fid not in factor_map:
                factor_map[fid] = f
            else:
                factor_map[fid]["mention_count"] += f["mention_count"]
                factor_map[fid]["sources"].extend(f["sources"])

        unique_factors = list(factor_map.values())
        for f in unique_factors:
            f["sources"] = list(set(f["sources"]))

        query = """
        UNWIND $factors AS factor
        MERGE (f:Factor {id: factor.id})
        SET f.name = factor.name,
            f.category = factor.category,
            f.mention_count = factor.mention_count,
            f.source_count = size(factor.sources)
        RETURN count(f) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, factors=unique_factors)
            factor_count = result.single()["count"]
            print(f"  Factor 노드 생성: {factor_count}개")

        # 관계 정규화 및 적재
        normalized_relations = []
        for rel in relations:
            factor_name = next(
                (f["name"] for f in factors if f["id"] == rel["factor_id"]),
                rel["factor_id"]
            )
            normalized_name = normalize_name(factor_name, reverse_mapping, exclude_list)
            if not normalized_name:
                continue

            normalized_relations.append({
                "factor_id": normalized_name.lower().replace(" ", "_"),
                "anchor_id": rel["anchor_id"],
                "relation_type": rel["relation_type"],
                "mention_count": rel.get("mention_count", 1)
            })

        # 관계 병합
        rel_map = {}
        for rel in normalized_relations:
            key = (rel["factor_id"], rel["anchor_id"], rel["relation_type"])
            if key not in rel_map:
                rel_map[key] = rel
            else:
                rel_map[key]["mention_count"] += rel["mention_count"]

        unique_relations = list(rel_map.values())

        query = """
        UNWIND $relations AS rel
        MATCH (f:Factor {id: rel.factor_id})
        MATCH (a:Anchor {id: rel.anchor_id})
        MERGE (f)-[r:AFFECTS {type: rel.relation_type}]->(a)
        SET r.mention_count = rel.mention_count
        RETURN count(r) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, relations=unique_relations)
            rel_count = result.single()["count"]
            print(f"  AFFECTS 관계 생성: {rel_count}개")

        return {"factors": factor_count, "relations": rel_count}

    def _rebuild_factor_factor(self, reverse_mapping, category_mapping, exclude_list) -> dict:
        """Factor-Factor 관계 재구축"""
        input_path = self.layer2_dir / "factor_relations.json"

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        relations = data.get("relations", [])

        # 정규화
        normalized = []
        new_factors = set()

        for rel in relations:
            source = normalize_name(rel["source_factor"], reverse_mapping, exclude_list)
            target = normalize_name(rel["target_factor"], reverse_mapping, exclude_list)

            if not source or not target:
                continue
            if source == target:
                continue

            normalized.append({
                "source_id": source.lower().replace(" ", "_"),
                "source_name": source,
                "target_id": target.lower().replace(" ", "_"),
                "target_name": target,
                "relation_type": rel["relation_type"],
                "mention_count": rel.get("mention_count", 1)
            })
            new_factors.add((source.lower().replace(" ", "_"), source, category_mapping.get(source, "extracted")))
            new_factors.add((target.lower().replace(" ", "_"), target, category_mapping.get(target, "extracted")))

        # 관계 병합
        rel_map = {}
        for rel in normalized:
            key = (rel["source_id"], rel["target_id"], rel["relation_type"])
            if key not in rel_map:
                rel_map[key] = rel
            else:
                rel_map[key]["mention_count"] += rel["mention_count"]

        unique_relations = list(rel_map.values())

        # 새 Factor 노드 생성 (없는 것만)
        new_factor_list = [{"id": f[0], "name": f[1], "category": f[2]} for f in new_factors]

        query = """
        UNWIND $factors AS factor
        MERGE (f:Factor {id: factor.id})
        ON CREATE SET f.name = factor.name,
                      f.category = factor.category,
                      f.mention_count = 0
        RETURN count(f) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, factors=new_factor_list)
            new_count = result.single()["count"]
            print(f"  새 Factor 노드 확인/생성: {new_count}개")

        # Factor-Factor 관계 생성
        query = """
        UNWIND $relations AS rel
        MATCH (source:Factor {id: rel.source_id})
        MATCH (target:Factor {id: rel.target_id})
        MERGE (source)-[r:INFLUENCES {type: rel.relation_type}]->(target)
        SET r.mention_count = rel.mention_count
        RETURN count(r) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, relations=unique_relations)
            rel_count = result.single()["count"]
            print(f"  INFLUENCES 관계 생성: {rel_count}개")

        return {"new_factors": len(new_factors), "relations": rel_count}

    def _verify(self) -> dict:
        """적재 결과 검증"""
        queries = {
            "factor_count": "MATCH (f:Factor) RETURN count(f) as count",
            "affects_count": "MATCH (:Factor)-[r:AFFECTS]->(:Anchor) RETURN count(r) as count",
            "influences_count": "MATCH (:Factor)-[r:INFLUENCES]->(:Factor) RETURN count(r) as count",
            "top_factors": """
                MATCH (f:Factor)
                WHERE f.mention_count > 0
                RETURN f.name as name, f.mention_count as mentions, f.category as category
                ORDER BY f.mention_count DESC
                LIMIT 10
            """,
            "top_affects": """
                MATCH (f:Factor)-[r:AFFECTS]->(a:Anchor)
                RETURN f.name as factor, r.type as type, a.name as anchor, r.mention_count as mentions
                ORDER BY r.mention_count DESC
                LIMIT 10
            """,
            "top_influences": """
                MATCH (s:Factor)-[r:INFLUENCES]->(t:Factor)
                RETURN s.name as source, r.type as type, t.name as target, r.mention_count as mentions
                ORDER BY r.mention_count DESC
                LIMIT 10
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

        # 결과 출력
        print(f"\n=== 적재 완료 ===")
        print(f"Factor 노드: {results['factor_count']}개")
        print(f"AFFECTS 관계: {results['affects_count']}개")
        print(f"INFLUENCES 관계: {results['influences_count']}개")

        print("\n--- 상위 Factor (언급 횟수) ---")
        for f in results.get("top_factors", []):
            print(f"  {f['name']}: {f['mentions']}회 [{f['category']}]")

        print("\n--- 상위 Factor→Anchor 관계 ---")
        for r in results.get("top_affects", []):
            print(f"  {r['factor']} --{r['type']}--> {r['anchor']}: {r['mentions']}회")

        print("\n--- 상위 Factor→Factor 관계 ---")
        for r in results.get("top_influences", []):
            arrow = "→유발→" if r['type'] == 'CAUSES' else ("→강화→" if r['type'] == 'AMPLIFIES' else "→완화→")
            print(f"  {r['source']} {arrow} {r['target']}: {r['mentions']}회")

        return results


def rebuild_layer2():
    """Layer 2 재구축 실행"""
    with Layer2Rebuilder() as rebuilder:
        return rebuilder.rebuild()


if __name__ == "__main__":
    rebuild_layer2()
