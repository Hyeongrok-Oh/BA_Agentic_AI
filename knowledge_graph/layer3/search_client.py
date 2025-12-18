"""검색 클라이언트 - Brave Search API 기반 뉴스 수집"""

import os
import time
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SearchResult:
    """검색 결과"""
    title: str
    link: str
    snippet: str
    date: Optional[str] = None
    source: Optional[str] = None
    query: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet,
            "date": self.date,
            "source": self.source,
            "query": self.query
        }


class BraveSearchClient:
    """Brave Search API 클라이언트"""

    BASE_URL = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY가 설정되지 않았습니다")

        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        self.request_count = 0
        self.last_request_time = None

    def search_news(
        self,
        query: str,
        count: int = 20,
        freshness: str = "pm"  # pd=past day, pw=past week, pm=past month, py=past year
    ) -> List[SearchResult]:
        """뉴스 검색"""
        url = f"{self.BASE_URL}/news/search"
        params = {
            "q": query,
            "count": min(count, 20),  # Brave는 최대 20개
            "freshness": freshness
        }

        self._rate_limit()

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            self.request_count += 1
            results = []

            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    link=item.get("url", ""),
                    snippet=item.get("description", ""),
                    date=item.get("age"),
                    source=item.get("meta_url", {}).get("hostname") if item.get("meta_url") else None,
                    query=query
                ))

            return results

        except requests.exceptions.RequestException as e:
            print(f"검색 오류 ({query}): {e}")
            return []

    def search_web(
        self,
        query: str,
        count: int = 10,
        freshness: str = "pm"
    ) -> List[SearchResult]:
        """일반 웹 검색"""
        url = f"{self.BASE_URL}/web/search"
        params = {
            "q": query,
            "count": min(count, 20),
            "freshness": freshness
        }

        self._rate_limit()

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            self.request_count += 1
            results = []

            for item in data.get("web", {}).get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    link=item.get("url", ""),
                    snippet=item.get("description", ""),
                    query=query
                ))

            return results

        except requests.exceptions.RequestException as e:
            print(f"검색 오류 ({query}): {e}")
            return []

    def _rate_limit(self, min_interval: float = 1.5):
        """Rate limiting (초당 1회 미만 제한)"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()


class QueryLoader:
    """검색 쿼리 로더"""

    def __init__(self, queries_path: Optional[Path] = None):
        self.queries_path = queries_path or Path(__file__).parent / "search_queries.yaml"
        self.queries = self._load_queries()

    def _load_queries(self) -> Dict:
        """YAML에서 쿼리 로드"""
        with open(self.queries_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_all_queries(self) -> List[str]:
        """모든 쿼리 반환"""
        all_queries = []

        # 카테고리별 쿼리
        for category, queries in self.queries.get("categories", {}).items():
            all_queries.extend(queries)

        # 지역별 쿼리
        for region, queries in self.queries.get("regions", {}).items():
            all_queries.extend(queries)

        # 제품별 쿼리
        all_queries.extend(self.queries.get("products", []))

        return all_queries

    def get_queries_by_category(self, category: str) -> List[str]:
        """카테고리별 쿼리"""
        return self.queries.get("categories", {}).get(category, [])

    def get_queries_by_region(self, region: str) -> List[str]:
        """지역별 쿼리"""
        return self.queries.get("regions", {}).get(region, [])

    @property
    def total_queries(self) -> int:
        return len(self.get_all_queries())


class NewsCollector:
    """뉴스 수집기"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        queries_path: Optional[Path] = None
    ):
        self.client = BraveSearchClient(api_key)
        self.query_loader = QueryLoader(queries_path)
        self.results: List[SearchResult] = []

    def collect_all(
        self,
        count_per_query: int = 20,
        freshness: str = "pm",
        verbose: bool = True
    ) -> List[SearchResult]:
        """모든 쿼리에 대해 뉴스 수집"""
        queries = self.query_loader.get_all_queries()
        total = len(queries)

        if verbose:
            print(f"=== 뉴스 수집 시작 ===")
            print(f"총 쿼리: {total}개")
            print(f"쿼리당 결과: {count_per_query}개")
            print(f"예상 총 결과: ~{total * count_per_query}개")
            print()

        self.results = []

        for i, query in enumerate(queries, 1):
            results = self.client.search_news(
                query=query,
                count=count_per_query,
                freshness=freshness
            )
            self.results.extend(results)

            if verbose and i % 10 == 0:
                print(f"  진행: {i}/{total} ({len(self.results)}개 수집)")

        # 중복 제거 (URL 기준)
        unique_results = self._deduplicate()

        if verbose:
            print(f"\n=== 수집 완료 ===")
            print(f"총 검색 결과: {len(self.results)}개")
            print(f"중복 제거 후: {len(unique_results)}개")
            print(f"API 호출 횟수: {self.client.request_count}회")

        self.results = unique_results
        return unique_results

    def collect_by_category(
        self,
        category: str,
        count_per_query: int = 20,
        freshness: str = "pm",
        verbose: bool = True
    ) -> List[SearchResult]:
        """카테고리별 뉴스 수집"""
        queries = self.query_loader.get_queries_by_category(category)

        if verbose:
            print(f"카테고리 '{category}' 검색: {len(queries)}개 쿼리")

        results = []
        for query in queries:
            query_results = self.client.search_news(
                query=query,
                count=count_per_query,
                freshness=freshness
            )
            results.extend(query_results)

        return self._deduplicate(results)

    def _deduplicate(self, results: Optional[List[SearchResult]] = None) -> List[SearchResult]:
        """URL 기준 중복 제거"""
        results = results or self.results
        seen_urls = set()
        unique = []

        for r in results:
            if r.link not in seen_urls:
                seen_urls.add(r.link)
                unique.append(r)

        return unique

    def save_results(self, output_path: Path) -> None:
        """결과 저장"""
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "collected_at": datetime.now().isoformat(),
                    "total_results": len(self.results),
                    "results": [r.to_dict() for r in self.results]
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"저장 완료: {output_path}")


def test_brave_search():
    """Brave Search API 테스트"""
    print("=== Brave Search API 테스트 ===\n")

    client = BraveSearchClient()

    # 테스트 쿼리
    test_queries = [
        "홍해 사태 해운",
        "LCD 패널 가격",
        "트럼프 관세 TV"
    ]

    for query in test_queries:
        print(f"쿼리: '{query}'")
        results = client.search_news(query, count=5, freshness="pm")
        print(f"  결과: {len(results)}개")
        for r in results[:2]:
            print(f"  - {r.title[:50]}...")
            print(f"    {r.date} | {r.source}")
        print()


if __name__ == "__main__":
    test_brave_search()
