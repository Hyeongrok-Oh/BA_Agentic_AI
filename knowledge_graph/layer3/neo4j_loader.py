"""Neo4j 로더 - Event 노드와 관계 적재"""

import os
from typing import List, Optional
from neo4j import GraphDatabase

from .models import EventNode, EventChunk, Layer3Graph
from .config import Layer3Config


class Layer3Neo4jLoader:
    """Layer 3 Neo4j 로더"""

    def __init__(self, config: Optional[Layer3Config] = None):
        self.config = config or Layer3Config()
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
        """Constraint 및 Index 설정"""
        queries = [
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX event_category IF NOT EXISTS FOR (e:Event) ON (e.category)",
            "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.start_date)",
            "CREATE INDEX event_ongoing IF NOT EXISTS FOR (e:Event) ON (e.is_ongoing)",
        ]

        with self.driver.session(database=self.config.neo4j_database) as session:
            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    print(f"  Constraint 설정 오류: {e}")

    def clear_layer3_data(self):
        """기존 Layer 3 데이터 삭제"""
        queries = [
            "MATCH (:Event)-[r:INCREASES]->(:Factor) DELETE r",
            "MATCH (:Event)-[r:DECREASES]->(:Factor) DELETE r",
            "MATCH (:Event)-[r:TARGETS]->(:Dimension) DELETE r",
            "MATCH (e:Event) DETACH DELETE e",
        ]

        with self.driver.session(database=self.config.neo4j_database) as session:
            for query in queries:
                result = session.run(query)
                summary = result.consume()
                deleted = summary.counters.relationships_deleted + summary.counters.nodes_deleted
                if deleted > 0:
                    print(f"  삭제: {deleted}개")

    def load_events(self, events: List[EventNode]) -> int:
        """Event 노드 적재"""
        query = """
        UNWIND $events AS event
        MERGE (e:Event {id: event.id})
        SET e.name = event.name,
            e.name_en = event.name_en,
            e.category = event.category,
            e.start_date = CASE WHEN event.start_date IS NOT NULL
                           THEN date(event.start_date) ELSE null END,
            e.end_date = CASE WHEN event.end_date IS NOT NULL
                         THEN date(event.end_date) ELSE null END,
            e.is_ongoing = event.is_ongoing,
            e.severity = event.severity,
            e.region_scope = event.region_scope,
            e.aliases = event.aliases,
            e.evidence = event.evidence,
            e.source_count = event.source_count,
            e.source_urls = event.source_urls,
            e.source_titles = event.source_titles
        RETURN count(e) as count
        """

        event_data = []
        for e in events:
            # 출처 URL과 제목 추출
            source_urls = [s.url for s in e.sources if s.url][:5]  # 최대 5개
            source_titles = [s.title for s in e.sources if s.title][:5]

            event_data.append({
                "id": e.id,
                "name": e.name,
                "name_en": e.name_en,
                "category": e.category.value,
                "start_date": e.start_date.isoformat() if e.start_date else None,
                "end_date": e.end_date.isoformat() if e.end_date else None,
                "is_ongoing": e.is_ongoing,
                "severity": e.severity.value,
                "region_scope": e.region_scope,
                "aliases": e.aliases,
                "evidence": e.evidence[:500] if e.evidence else "",  # 최대 500자
                "source_count": e.source_count,
                "source_urls": source_urls,
                "source_titles": source_titles
            })

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, events=event_data)
            return result.single()["count"]

    def load_event_factor_relations(self, events: List[EventNode]) -> dict:
        """Event → Factor 관계 적재"""
        increases_query = """
        UNWIND $relations AS rel
        MATCH (e:Event {id: rel.event_id})
        MATCH (f:Factor {id: rel.factor_id})
        MERGE (e)-[r:INCREASES]->(f)
        SET r.magnitude = rel.magnitude,
            r.confidence = rel.confidence,
            r.evidence = rel.evidence
        RETURN count(r) as count
        """

        decreases_query = """
        UNWIND $relations AS rel
        MATCH (e:Event {id: rel.event_id})
        MATCH (f:Factor {id: rel.factor_id})
        MERGE (e)-[r:DECREASES]->(f)
        SET r.magnitude = rel.magnitude,
            r.confidence = rel.confidence,
            r.evidence = rel.evidence
        RETURN count(r) as count
        """

        increases = []
        decreases = []

        for event in events:
            for rel in event.factor_relations:
                rel_data = {
                    "event_id": event.id,
                    "factor_id": rel.factor_id,
                    "magnitude": rel.magnitude,
                    "confidence": rel.confidence,
                    "evidence": rel.evidence[:200] if rel.evidence else ""
                }

                if rel.impact_type.value == "INCREASES":
                    increases.append(rel_data)
                else:
                    decreases.append(rel_data)

        counts = {"INCREASES": 0, "DECREASES": 0}

        with self.driver.session(database=self.config.neo4j_database) as session:
            if increases:
                result = session.run(increases_query, relations=increases)
                counts["INCREASES"] = result.single()["count"]

            if decreases:
                result = session.run(decreases_query, relations=decreases)
                counts["DECREASES"] = result.single()["count"]

        return counts

    def load_event_dimension_relations(self, events: List[EventNode]) -> int:
        """Event → Dimension 관계 적재"""
        query = """
        UNWIND $relations AS rel
        MATCH (e:Event {id: rel.event_id})
        MATCH (d:Dimension {name: rel.dimension_name})
        MERGE (e)-[r:TARGETS]->(d)
        SET r.specificity = rel.specificity
        RETURN count(r) as count
        """

        relations = []
        for event in events:
            for rel in event.dimension_relations:
                relations.append({
                    "event_id": event.id,
                    "dimension_name": rel.dimension_name,
                    "specificity": rel.specificity
                })

        if not relations:
            return 0

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, relations=relations)
            return result.single()["count"]

    def load_embeddings(self, chunks: List[EventChunk]) -> int:
        """Vector Embedding을 Event 노드에 저장"""
        # 청크별 임베딩을 Event에 저장 (첫 번째 청크만)
        query = """
        UNWIND $embeddings AS emb
        MATCH (e:Event {id: emb.event_id})
        SET e.embedding = emb.embedding
        RETURN count(e) as count
        """

        # Event별 첫 번째 청크만 선택
        event_embeddings = {}
        for chunk in chunks:
            if chunk.embedding and chunk.event_id not in event_embeddings:
                event_embeddings[chunk.event_id] = chunk.embedding

        if not event_embeddings:
            return 0

        embeddings_data = [
            {"event_id": eid, "embedding": emb}
            for eid, emb in event_embeddings.items()
        ]

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, embeddings=embeddings_data)
            return result.single()["count"]

    def verify_load(self) -> dict:
        """적재 결과 검증"""
        queries = {
            "event_count": "MATCH (e:Event) RETURN count(e) as count",
            "increases_count": "MATCH (:Event)-[r:INCREASES]->(:Factor) RETURN count(r) as count",
            "decreases_count": "MATCH (:Event)-[r:DECREASES]->(:Factor) RETURN count(r) as count",
            "targets_count": "MATCH (:Event)-[r:TARGETS]->(:Dimension) RETURN count(r) as count",
            "top_events": """
                MATCH (e:Event)
                OPTIONAL MATCH (e)-[r:INCREASES|DECREASES]->(:Factor)
                RETURN e.name as name, e.category as category,
                       e.severity as severity, count(r) as factor_count
                ORDER BY factor_count DESC
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

        return results


def load_layer3_to_neo4j(
    graph: Layer3Graph,
    clear_existing: bool = True,
    verbose: bool = True
) -> dict:
    """Layer 3 그래프를 Neo4j에 적재"""
    with Layer3Neo4jLoader() as loader:
        if verbose:
            print("=== Layer 3 Neo4j 적재 ===\n")

        # Constraint 설정
        if verbose:
            print("1. Constraint 설정...")
        loader.setup_constraints()

        # 기존 데이터 삭제
        if clear_existing:
            if verbose:
                print("\n2. 기존 데이터 삭제...")
            loader.clear_layer3_data()

        # Event 노드 적재
        if verbose:
            print(f"\n3. Event 노드 적재 ({len(graph.events)}개)...")
        event_count = loader.load_events(graph.events)
        if verbose:
            print(f"  적재: {event_count}개")

        # Event → Factor 관계 적재
        if verbose:
            print("\n4. Event → Factor 관계 적재...")
        factor_counts = loader.load_event_factor_relations(graph.events)
        if verbose:
            print(f"  INCREASES: {factor_counts['INCREASES']}개")
            print(f"  DECREASES: {factor_counts['DECREASES']}개")

        # Event → Dimension 관계 적재
        if verbose:
            print("\n5. Event → Dimension 관계 적재...")
        dim_count = loader.load_event_dimension_relations(graph.events)
        if verbose:
            print(f"  TARGETS: {dim_count}개")

        # Vector Embedding 적재
        if graph.chunks:
            if verbose:
                print(f"\n6. Vector Embedding 적재 ({len(graph.chunks)}개 청크)...")
            emb_count = loader.load_embeddings(graph.chunks)
            if verbose:
                print(f"  적재: {emb_count}개 Event에 임베딩 추가")

        # 검증
        if verbose:
            print("\n=== 적재 결과 ===")
        verification = loader.verify_load()

        if verbose:
            print(f"Event 노드: {verification['event_count']}개")
            print(f"INCREASES 관계: {verification['increases_count']}개")
            print(f"DECREASES 관계: {verification['decreases_count']}개")
            print(f"TARGETS 관계: {verification['targets_count']}개")

            print("\n--- 상위 Event ---")
            for e in verification.get("top_events", []):
                print(f"  {e['name']} [{e['category']}] - Factor: {e['factor_count']}개")

        return verification
