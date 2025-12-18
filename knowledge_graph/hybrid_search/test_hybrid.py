#!/usr/bin/env python3
"""
Hybrid Search Engine 테스트
"""

import os
import sys

# 경로 설정
sys.path.insert(0, '/Users/hyeongrokoh/BI/knowledge_graph')
sys.path.insert(0, '/Users/hyeongrokoh/BI/sql')

from hybrid_search import HybridSearchEngine, run_analysis


def test_full_engine():
    """전체 엔진 테스트"""
    print("=" * 60)
    print("Hybrid Search Engine 전체 테스트")
    print("=" * 60)

    # 테스트 질문
    question = "2024년 4분기 북미 영업이익이 왜 감소했어?"

    # 엔진 실행
    result = run_analysis(
        question=question,
        year=2024,
        quarter=4,
        region="NA"
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("상세 분석 결과")
    print("=" * 60)

    print(f"\n질문: {result.question}")
    print(f"생성된 가설 수: {len(result.hypotheses)}")
    print(f"검증된 가설 수: {len(result.validated_hypotheses)}")

    if result.validated_hypotheses:
        print("\n검증된 가설 상세:")
        for h in result.validated_hypotheses:
            data = h.validation_data or {}
            print(f"\n  [{h.id}] {h.factor}")
            print(f"    - 설명: {h.description}")
            print(f"    - 변화율: {data.get('change_percent', 0):+.1f}%")
            print(f"    - 이전값: {data.get('previous_value', 0):,.0f}")
            print(f"    - 현재값: {data.get('current_value', 0):,.0f}")

    if result.graph_evidences:
        print("\n관련 외부 이벤트:")
        for h_id, evidences in result.graph_evidences.items():
            print(f"\n  [{h_id}]")
            for ev in evidences[:3]:
                print(f"    - {ev.event_name} ({ev.event_category})")

    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(result.summary)

    return result


def test_hypothesis_generator():
    """가설 생성기 단독 테스트"""
    from hybrid_search.hypothesis_generator import HypothesisGenerator

    print("=" * 60)
    print("가설 생성기 테스트")
    print("=" * 60)

    generator = HypothesisGenerator()

    hypotheses = generator.generate(
        question="2024년 4분기 영업이익 하락 원인 분석",
        company="LGE",
        period="2024년 Q4",
        region="NA"
    )

    print(f"\n생성된 가설: {len(hypotheses)}개")
    for h in hypotheses:
        print(f"\n  [{h.id}] {h.category}")
        print(f"    Factor: {h.factor}")
        print(f"    Direction: {h.direction}")
        print(f"    Description: {h.description}")

    return hypotheses


def test_hypothesis_validator(hypotheses=None):
    """가설 검증기 단독 테스트"""
    from hybrid_search.hypothesis_validator import HypothesisValidator
    from hybrid_search.hypothesis_generator import HypothesisGenerator, Hypothesis

    print("=" * 60)
    print("가설 검증기 테스트")
    print("=" * 60)

    # 가설이 없으면 샘플 생성
    if not hypotheses:
        hypotheses = [
            Hypothesis(
                id="H1",
                category="cost",
                factor="물류비",
                direction="increase",
                description="물류비 증가로 인한 영업이익 감소",
                sql_check="COST_TYPE = 'LOG' 비교"
            ),
            Hypothesis(
                id="H2",
                category="cost",
                factor="재료비",
                direction="increase",
                description="재료비 증가로 인한 원가 상승",
                sql_check="COST_TYPE = 'MAT' 비교"
            ),
            Hypothesis(
                id="H3",
                category="pricing",
                factor="Price Protection",
                direction="increase",
                description="프로모션 비용 증가로 인한 수익성 악화",
                sql_check="COND_TYPE = 'ZPRO' 비교"
            )
        ]

    validator = HypothesisValidator()

    validated = validator.validate_hypotheses(
        hypotheses=hypotheses,
        period={"year": 2024, "quarter": 4},
        region="NA",
        threshold=5.0
    )

    print(f"\n검증 결과: {len(validated)}/{len(hypotheses)} 가설 검증됨")
    for h in validated:
        data = h.validation_data or {}
        print(f"\n  [{h.id}] {h.factor}")
        print(f"    변화율: {data.get('change_percent', 0):+.1f}%")
        print(f"    방향: {data.get('direction', '')}")

    return validated


def test_graph_searcher(validated=None):
    """그래프 검색기 단독 테스트"""
    from hybrid_search.graph_searcher import GraphSearcher
    from hybrid_search.hypothesis_generator import Hypothesis

    print("=" * 60)
    print("Graph 검색기 테스트")
    print("=" * 60)

    # 검증된 가설이 없으면 샘플 생성
    if not validated:
        validated = [
            Hypothesis(
                id="H1",
                category="cost",
                factor="물류비",
                direction="increase",
                description="물류비 증가",
                sql_check="",
                validated=True
            )
        ]

    try:
        searcher = GraphSearcher()

        evidences = searcher.search_for_hypotheses(
            hypotheses=validated,
            region="NA"
        )

        print(f"\n검색된 이벤트:")
        for h_id, evs in evidences.items():
            print(f"\n  [{h_id}] {len(evs)}개 이벤트")
            for ev in evs[:5]:
                print(f"    - {ev.event_name}")
                print(f"      카테고리: {ev.event_category}")
                print(f"      영향: {ev.impact_type} {ev.factor_name}")

        searcher.close()
        return evidences

    except Exception as e:
        print(f"\n⚠️ Graph 검색 오류: {e}")
        print("  Neo4j 연결을 확인하세요.")
        return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Search 테스트")
    parser.add_argument("--component", choices=["generator", "validator", "graph", "full"],
                        default="full", help="테스트할 컴포넌트")

    args = parser.parse_args()

    if args.component == "generator":
        test_hypothesis_generator()
    elif args.component == "validator":
        test_hypothesis_validator()
    elif args.component == "graph":
        test_graph_searcher()
    else:
        test_full_engine()
