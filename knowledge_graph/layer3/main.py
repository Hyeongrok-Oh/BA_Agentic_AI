"""Layer 3 메인 실행 파일"""

import json
import argparse
from pathlib import Path
from datetime import datetime

from .search_client import NewsCollector
from .event_extractor import Layer3Builder
from .normalizer import normalize_layer3
from .vector_store import process_layer3_vectors
from .neo4j_loader import load_layer3_to_neo4j
from .models import Layer3Graph


def build_layer3(
    max_queries: int = 60,
    results_per_query: int = 20,
    freshness: str = "pm",  # past month
    skip_vectors: bool = False,
    verbose: bool = True
) -> Layer3Graph:
    """Layer 3 전체 파이프라인 실행"""
    layer3_dir = Path(__file__).parent

    # 1. 뉴스 수집
    if verbose:
        print("=" * 50)
        print("Phase 1: 뉴스 수집")
        print("=" * 50)

    collector = NewsCollector()
    all_queries = collector.query_loader.get_all_queries()[:max_queries]

    # 쿼리 제한
    collector.query_loader.queries["categories"] = {
        k: v[:max_queries // 6] for k, v in
        collector.query_loader.queries.get("categories", {}).items()
    }

    search_results = collector.collect_all(
        count_per_query=results_per_query,
        freshness=freshness,
        verbose=verbose
    )

    # 중간 결과 저장
    search_output = layer3_dir / "search_results.json"
    collector.save_results(search_output)

    # 2. Event 추출
    if verbose:
        print("\n" + "=" * 50)
        print("Phase 2: Event 추출")
        print("=" * 50)

    builder = Layer3Builder()
    graph = builder.build_from_search_results(
        search_results,
        batch_size=10,
        verbose=verbose
    )

    # 3. 정규화
    if verbose:
        print("\n" + "=" * 50)
        print("Phase 3: Event 정규화")
        print("=" * 50)

    graph = normalize_layer3(graph)

    # 4. Vector 처리
    if not skip_vectors:
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 4: Vector Embedding 생성")
            print("=" * 50)

        graph = process_layer3_vectors(graph, verbose=verbose)

    # 결과 저장
    output_path = layer3_dir / "layer3_result.json"
    save_layer3_result(graph, output_path)

    return graph


def save_layer3_result(graph: Layer3Graph, output_path: Path) -> None:
    """Layer 3 결과 저장"""
    result = {
        "generated_at": datetime.now().isoformat(),
        "summary": graph.summary(),
        "events": [e.to_dict() for e in graph.events],
        "chunks": [c.to_dict() for c in graph.chunks] if graph.chunks else []
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_path}")


def load_and_process(input_path: Path, skip_vectors: bool = False) -> Layer3Graph:
    """저장된 결과 로드 및 처리"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Layer3Graph 재구성
    from .models import (
        EventNode, EventCategory, Severity, ImpactType,
        EventFactorRelation, EventDimensionRelation, EventSource
    )

    graph = Layer3Graph()

    for e_data in data.get("events", []):
        # Factor 관계 파싱
        factor_relations = []
        for fr in e_data.get("factor_relations", []):
            factor_relations.append(EventFactorRelation(
                factor_name=fr.get("factor", ""),
                factor_id=fr.get("factor", "").lower().replace(" ", "_"),
                impact_type=ImpactType(fr.get("impact", "INCREASES")),
                magnitude=fr.get("magnitude", "medium")
            ))

        # Dimension 관계 파싱
        dimension_relations = []
        for dr in e_data.get("dimension_relations", []):
            dimension_relations.append(EventDimensionRelation(
                dimension_name=dr.get("dimension", ""),
                dimension_type=dr.get("type", "Region")
            ))

        event = EventNode(
            id=e_data.get("id", ""),
            name=e_data.get("name", ""),
            name_en=e_data.get("name_en"),
            category=EventCategory(e_data.get("category", "market")),
            severity=Severity(e_data.get("severity", "medium")),
            is_ongoing=e_data.get("is_ongoing", False),
            factor_relations=factor_relations,
            dimension_relations=dimension_relations,
            evidence=e_data.get("evidence", "")
        )

        graph.events.append(event)

    # Vector 처리
    if not skip_vectors and not graph.chunks:
        graph = process_layer3_vectors(graph, verbose=True)

    return graph


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description="Layer 3 Event 추출 및 적재")
    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # build 명령어
    build_parser = subparsers.add_parser("build", help="뉴스 수집 및 Event 추출")
    build_parser.add_argument("--max-queries", type=int, default=60, help="최대 쿼리 수")
    build_parser.add_argument("--results-per-query", type=int, default=20, help="쿼리당 결과 수")
    build_parser.add_argument("--skip-vectors", action="store_true", help="Vector 생성 건너뛰기")

    # load 명령어
    load_parser = subparsers.add_parser("load", help="Neo4j에 적재")
    load_parser.add_argument("--input", type=str, default="layer3_result.json", help="입력 파일")
    load_parser.add_argument("--no-clear", action="store_true", help="기존 데이터 유지")

    # full 명령어 (build + load)
    full_parser = subparsers.add_parser("full", help="전체 파이프라인 (build + load)")
    full_parser.add_argument("--max-queries", type=int, default=60, help="최대 쿼리 수")
    full_parser.add_argument("--results-per-query", type=int, default=20, help="쿼리당 결과 수")
    full_parser.add_argument("--skip-vectors", action="store_true", help="Vector 생성 건너뛰기")

    args = parser.parse_args()

    if args.command == "build":
        graph = build_layer3(
            max_queries=args.max_queries,
            results_per_query=args.results_per_query,
            skip_vectors=args.skip_vectors
        )
        print(f"\n빌드 완료: {graph.summary()}")

    elif args.command == "load":
        layer3_dir = Path(__file__).parent
        input_path = layer3_dir / args.input

        if not input_path.exists():
            print(f"파일 없음: {input_path}")
            return

        graph = load_and_process(input_path)
        load_layer3_to_neo4j(graph, clear_existing=not args.no_clear)

    elif args.command == "full":
        # Build
        graph = build_layer3(
            max_queries=args.max_queries,
            results_per_query=args.results_per_query,
            skip_vectors=args.skip_vectors
        )

        # Load
        load_layer3_to_neo4j(graph, clear_existing=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
