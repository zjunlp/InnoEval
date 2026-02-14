"""
Web Searcher Module for V2

Searches the web using the Serper API.
"""

import logging
import http.client
import json
import os
from typing import List, Optional, Tuple

from .models import Source, SourceType, Platform

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    Searches the web using Serper API.
    """
    
    def __init__(self, max_results_per_query: int = 3):
        """
        Initialize the web searcher.
        
        Args:
            max_results_per_query: Maximum results per query (default: 3)
        """
        self.max_results = max_results_per_query or 3
        self.api_key = os.getenv("SERPER_KEY_ID")
        
        if not self.api_key:
            logger.warning("Serper API key not found. Web search will be disabled.")
        
        logger.info(f"Initialized WebSearcher with max_results: {self.max_results}")
    
    def search(self, queries: List[str]) -> List[Tuple[Source, int]]:
        """
        Search the web using multiple queries.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of (Source, query_index) tuples
        """
        if not self.api_key:
            logger.warning("Cannot perform web search: API key not configured")
            return []
        
        all_results: List[Tuple[Source, int]] = []
        
        for q_idx, query in enumerate(queries):
            logger.info(f"Searching web for query[{q_idx}]: {query}")
            try:
                results = self._google_search_debug_api(query)
                for src in results:
                    if src.metadata is None:
                        src.metadata = {}
                    src.metadata["query_index"] = q_idx
                    src.metadata["query"] = query
                    all_results.append((src, q_idx))
            except Exception as e:
                logger.error(f"Error searching web for query '{query}': {e}")
                continue
        
        # Deduplicate by URL
        unique_results = self._deduplicate(all_results)
        logger.info(f"Found {len(unique_results)} unique web results with indices")
        
        return unique_results
    
    def _google_search_debug_api(self, query: str) -> List[Source]:
        """
        Perform Google search using Serper API.
        
        Args:
            query: Search query
            
        Returns:
            List of Source objects
        """
        conn = http.client.HTTPSConnection("google.serper.dev")
        contains_chinese = any('\u4E00' <= char <= '\u9FFF' for char in query)
        
        # Add site restrictions
        query += " AND (site:x.com OR site:medium.com OR site:towardsdatascience.com OR site:substack.com OR site:reddit.com/r/MachineLearning)"
        
        payload = json.dumps({
            "q": query,
            "location": "China" if contains_chinese else "United States",
            "gl": "cn" if contains_chinese else "us",
            "hl": "zh-cn" if contains_chinese else "en",
            "num": self.max_results,
        })
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
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
                title = page.get("title", "Untitled")
                link = page.get("link", "")
                snippet = page.get("snippet", "")
                date = page.get("date", "")
                source_name = page.get("source", "")
                description = " | ".join([p for p in [snippet, f"Published: {date}" if date else None, f"Source: {source_name}" if source_name else None] if p])
                s = Source(
                    title=title,
                    url=link,
                    source_type=SourceType.WEBPAGE,
                    platform=Platform.GOOGLE_SEARCH,
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

