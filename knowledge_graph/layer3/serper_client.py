"""Serper API 클라이언트 - Google 검색 기반 뉴스 수집"""

import os
import time
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, date
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


class SerperClient:
    """Serper API 클라이언트"""

    BASE_URL = "https://google.serper.dev"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY가 설정되지 않았습니다")

        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        self.request_count = 0
        self.last_request_time = None

    def search_news(
        self,
        query: str,
        num_results: int = 20,
        country: str = "kr",
        language: str = "ko"
    ) -> List[SearchResult]:
        """뉴스 검색"""
        url = f"{self.BASE_URL}/news"
        payload = {
            "q": query,
            "gl": country,
            "hl": language,
            "num": num_results
        }

        # Rate limiting (초당 1회)
        self._rate_limit()

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            self.request_count += 1
            results = []

            for item in data.get("news", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    date=item.get("date"),
                    source=item.get("source"),
                    query=query
                ))

            return results

        except requests.exceptions.RequestException as e:
            print(f"검색 오류 ({query}): {e}")
            return []

    def search_web(
        self,
        query: str,
        num_results: int = 10,
        country: str = "kr",
        language: str = "ko"
    ) -> List[SearchResult]:
        """일반 웹 검색"""
        url = f"{self.BASE_URL}/search"
        payload = {
            "q": query,
            "gl": country,
            "hl": language,
            "num": num_results
        }

        self._rate_limit()

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            self.request_count += 1
            results = []

            for item in data.get("organic", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    query=query
                ))

            return results

        except requests.exceptions.RequestException as e:
            print(f"검색 오류 ({query}): {e}")
            return []

    def _rate_limit(self, min_interval: float = 0.5):
        """Rate limiting"""
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


class NewsCollector:
    """뉴스 수집기"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        queries_path: Optional[Path] = None
    ):
        self.client = SerperClient(api_key)
        self.query_loader = QueryLoader(queries_path)
        self.results: List[SearchResult] = []

    def collect_all(
        self,
        num_results_per_query: int = 20,
        languages: List[str] = ["ko", "en"],
        verbose: bool = True
    ) -> List[SearchResult]:
        """모든 쿼리에 대해 뉴스 수집"""
        queries = self.query_loader.get_all_queries()
        total = len(queries) * len(languages)

        if verbose:
            print(f"=== 뉴스 수집 시작 ===")
            print(f"총 쿼리: {len(queries)}개 × {len(languages)} 언어 = {total}회 검색")

        self.results = []
        count = 0

        for query in queries:
            for lang in languages:
                country = "kr" if lang == "ko" else "us"
                results = self.client.search_news(
                    query=query,
                    num_results=num_results_per_query,
                    country=country,
                    language=lang
                )
                self.results.extend(results)
                count += 1

                if verbose and count % 10 == 0:
                    print(f"  진행: {count}/{total} ({len(self.results)}개 수집)")

        # 중복 제거 (URL 기준)
        unique_results = self._deduplicate()

        if verbose:
            print(f"\n=== 수집 완료 ===")
            print(f"총 검색 결과: {len(self.results)}개")
            print(f"중복 제거 후: {len(unique_results)}개")
            print(f"API 호출 횟수: {self.client.request_count}회")

        return unique_results

    def collect_by_category(
        self,
        category: str,
        num_results_per_query: int = 20,
        language: str = "ko"
    ) -> List[SearchResult]:
        """카테고리별 뉴스 수집"""
        queries = self.query_loader.get_queries_by_category(category)
        results = []

        for query in queries:
            country = "kr" if language == "ko" else "us"
            query_results = self.client.search_news(
                query=query,
                num_results=num_results_per_query,
                country=country,
                language=language
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

    def to_dict(self) -> List[Dict]:
        """결과를 딕셔너리로 변환"""
        return [
            {
                "title": r.title,
                "link": r.link,
                "snippet": r.snippet,
                "date": r.date,
                "source": r.source,
                "query": r.query
            }
            for r in self.results
        ]


def test_serper():
    """Serper API 테스트"""
    print("=== Serper API 테스트 ===\n")

    client = SerperClient()

    # 단일 쿼리 테스트
    print("1. 단일 쿼리 테스트: '홍해 사태 해운 2025'")
    results = client.search_news("홍해 사태 해운 2025", num_results=5)
    print(f"   결과: {len(results)}개")
    for r in results[:3]:
        print(f"   - {r.title[:50]}...")

    print()

    # 영어 쿼리 테스트
    print("2. 영어 쿼리 테스트: 'Trump tariff TV 2025'")
    results = client.search_news(
        "Trump tariff TV 2025",
        num_results=5,
        country="us",
        language="en"
    )
    print(f"   결과: {len(results)}개")
    for r in results[:3]:
        print(f"   - {r.title[:50]}...")


if __name__ == "__main__":
    test_serper()
