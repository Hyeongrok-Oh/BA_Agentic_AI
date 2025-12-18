"""Layer 2 Neo4j 로더 - Factor 노드 및 관계 적재"""

import json
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from knowledge_graph.config import BaseConfig


class Layer2Neo4jLoader:
    """Layer 2 데이터를 Neo4j에 적재"""

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

    def setup_constraints(self):
        """Factor 노드 제약조건 설정"""
        constraints = [
            "CREATE CONSTRAINT factor_id IF NOT EXISTS FOR (f:Factor) REQUIRE f.id IS UNIQUE",
        ]

        with self.driver.session(database=self.config.neo4j_database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint 설정 중 오류 (무시): {e}")

    def clear_layer2_data(self):
        """기존 Layer 2 데이터 삭제"""
        queries = [
            # Factor → Anchor 관계 삭제
            "MATCH (:Factor)-[r:AFFECTS]->(:Anchor) DELETE r",
            # Factor 노드 삭제
            "MATCH (f:Factor) DELETE f",
        ]

        with self.driver.session(database=self.config.neo4j_database) as session:
            for query in queries:
                result = session.run(query)
                summary = result.consume()
                print(f"  삭제: {summary.counters.nodes_deleted} 노드, {summary.counters.relationships_deleted} 관계")

    def load_factors(self, factors: list) -> int:
        """Factor 노드 생성"""
        query = """
        UNWIND $factors AS factor
        MERGE (f:Factor {id: factor.id})
        SET f.name = factor.name,
            f.category = factor.category,
            f.mention_count = factor.mention_count,
            f.original_names = factor.original_names,
            f.source_count = size(factor.sources)
        RETURN count(f) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, factors=factors)
            record = result.single()
            return record["count"] if record else 0

    def load_relations(self, relations: list) -> int:
        """Factor → Anchor 관계 생성"""
        # Anchor ID는 이미 revenue, quantity, cost로 되어 있음
        query = """
        UNWIND $relations AS rel
        MATCH (f:Factor {id: rel.factor_id})
        MATCH (a:Anchor {id: rel.anchor_id})
        MERGE (f)-[r:AFFECTS {type: rel.relation_type}]->(a)
        SET r.mention_count = rel.mention_count,
            r.source_count = rel.source_count
        RETURN count(r) as count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, relations=relations)
            record = result.single()
            return record["count"] if record else 0

    def verify_load(self) -> dict:
        """적재 결과 검증"""
        queries = {
            "factor_count": "MATCH (f:Factor) RETURN count(f) as count",
            "relation_count": "MATCH (:Factor)-[r:AFFECTS]->(:Anchor) RETURN count(r) as count",
            "top_factors": """
                MATCH (f:Factor)
                RETURN f.name as name, f.mention_count as mentions
                ORDER BY f.mention_count DESC
                LIMIT 5
            """,
            "top_relations": """
                MATCH (f:Factor)-[r:AFFECTS]->(a:Anchor)
                RETURN f.name as factor, a.name as anchor, r.type as type, r.mention_count as mentions
                ORDER BY r.mention_count DESC
                LIMIT 5
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


def load_layer2_to_neo4j(
    input_file: str = "layer2_normalized.json",
    clear_existing: bool = True
):
    """Layer 2 정규화 데이터를 Neo4j에 적재"""
    layer2_dir = Path(__file__).parent
    input_path = layer2_dir / input_file

    # 데이터 로드
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    factors = data.get("factors", [])
    relations = data.get("relations", [])

    print("=== Layer 2 Neo4j 적재 시작 ===")
    print(f"입력 파일: {input_path}")
    print(f"Factor 수: {len(factors)}")
    print(f"관계 수: {len(relations)}")

    with Layer2Neo4jLoader() as loader:
        # 제약조건 설정
        print("\n1. 제약조건 설정...")
        loader.setup_constraints()

        # 기존 데이터 삭제
        if clear_existing:
            print("\n2. 기존 Layer 2 데이터 삭제...")
            loader.clear_layer2_data()

        # Factor 노드 생성
        print("\n3. Factor 노드 생성...")
        factor_count = loader.load_factors(factors)
        print(f"  생성된 Factor: {factor_count}개")

        # 관계 생성
        print("\n4. Factor → Anchor 관계 생성...")
        relation_count = loader.load_relations(relations)
        print(f"  생성된 관계: {relation_count}개")

        # 검증
        print("\n5. 적재 결과 검증...")
        verification = loader.verify_load()

        print(f"\n=== 적재 완료 ===")
        print(f"Factor 노드: {verification['factor_count']}개")
        print(f"AFFECTS 관계: {verification['relation_count']}개")

        print("\n--- 상위 Factor ---")
        for f in verification.get("top_factors", []):
            print(f"  {f['name']}: {f['mentions']}회")

        print("\n--- 상위 관계 ---")
        for r in verification.get("top_relations", []):
            print(f"  {r['factor']} --{r['type']}--> {r['anchor']}: {r['mentions']}회")

    return verification


if __name__ == "__main__":
    load_layer2_to_neo4j()
