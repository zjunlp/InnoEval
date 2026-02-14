"""
GitHub Web Searcher Module for V2

Searches GitHub repositories using the Serper API via Google search.
"""

import logging
import http.client
import json
import os
from typing import List, Tuple

from .models import Source, SourceType, Platform

logger = logging.getLogger(__name__)


class GithubWebSearcher:
    """
    Searches GitHub repositories using Serper API (Google search).
    """

    def __init__(self, max_results_per_query: int = 3):
        """
        Initialize the GitHub web searcher.

        Args:
            max_results_per_query: Maximum results per query (default: 3)
        """
        self.max_results = max_results_per_query or 3
        self.api_key = os.getenv("SERPER_KEY_ID")

        if not self.api_key:
            logger.warning("Serper API key not found. GitHub web search will be disabled.")

        logger.info(f"Initialized GithubWebSearcher with max_results: {self.max_results}")

    def search(self, queries: List[str]) -> List[Tuple[Source, int]]:
        """
        Search GitHub repositories using multiple queries.

        Args:
            queries: List of search queries

        Returns:
            List of tuples (Source, query_index)
        """
        if not self.api_key:
            logger.warning("Cannot perform GitHub web search: API key not configured")
            return []

        all_results: List[tuple] = []

        for q_idx, query in enumerate(queries):
            logger.info(f"Searching GitHub via web for query[{q_idx}]: {query}")
            try:
                results = self._google_search_debug_api(query)
                for src in results:
                    if src.metadata is None:
                        src.metadata = {}
                    src.metadata["query_index"] = q_idx
                    src.metadata["query"] = query
                    all_results.append((src, q_idx))
            except Exception as e:
                logger.error(f"Error searching GitHub via web for query '{query}': {e}")
                continue

        unique_results = self._deduplicate(all_results)
        logger.info(f"Found {len(unique_results)} unique GitHub web results with indices")

        return unique_results

    def _google_search_debug_api(self, query: str) -> List[Source]:
        """
        Perform Google search (Serper) restricted to GitHub.

        Args:
            query: Search query

        Returns:
            List of Source objects
        """
        conn = http.client.HTTPSConnection("google.serper.dev")
        contains_chinese = any("\u4E00" <= char <= "\u9FFF" for char in query)

        # Restrict results to GitHub
        # query += " AND site:github.com"

        payload = json.dumps(
            {
                "q": query,
                "location": "China" if contains_chinese else "United States",
                "gl": "cn" if contains_chinese else "us",
                "hl": "zh-cn" if contains_chinese else "en",
                "num": self.max_results,
            }
        )
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        # Retry logic
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                logger.warning(f"Attempt {i+1}/5 failed: {e}")
                if i == 4:
                    logger.error("Google search timeout after 5 attempts")
                    return []
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))
        sources: List[Source] = []

        try:
            if "organic" not in results:
                return []
            for idx, page in enumerate(results["organic"], 1):
                title = page.get("title", "Untitled Repository")
                link = page.get("link", "")
                snippet = page.get("snippet", "")
                date = page.get("date", "")
                source_name = page.get("source", "")
                description = " | ".join(
                    [
                        p
                        for p in [
                            snippet,
                            f"Published: {date}" if date else None,
                            f"Source: {source_name}" if source_name else None,
                        ]
                        if p
                    ]
                )
                s = Source(
                    title=title,
                    url=link,
                    source_type=SourceType.CODE,
                    platform=Platform.GITHUB,
                    description=description,
                    metadata={"date": date, "source": source_name, "snippet": snippet, "rank": idx},
                    timestamp=date or None,
                    web_source_name=source_name or None,
                    web_rank=idx,
                )
                sources.append(s)
        except Exception as e:
            logger.error(f"Error parsing Google search results: {e}")

        return sources

    def _deduplicate(self, sources: List[Tuple[Source, int]]) -> List[Tuple[Source, int]]:
        """
        Remove duplicate sources based on URL.

        Args:
            sources: List of (Source, query_index) tuples

        Returns:
            Deduplicated list of (Source, query_index) tuples
        """
        seen_urls = set()
        unique_sources = []

        for source, q_idx in sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append((source, q_idx))

        return unique_sources

