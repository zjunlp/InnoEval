"""
Paper Searcher Module for V2

Searches for academic papers on arXiv with filtering support.
"""

import logging
import re
import requests
import json
import os
import dspy
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup

from .models import Source, SourceType, Platform

logger = logging.getLogger(__name__)


def parse_arxiv_xml(xml_data: str) -> list:
    """Parse arXiv XML response into paper dictionaries."""
    papers = []
    soup = BeautifulSoup(xml_data, "xml")
    
    for entry in soup.find_all("entry"):
        try:
            # Title
            title_elem = entry.find("title")
            title_text = title_elem.text.strip() if title_elem else ""
            
            # Abstract
            summary_elem = entry.find("summary")
            abstract_text = summary_elem.text.strip() if summary_elem else ""
            
            # Authors
            authors = []
            for author in entry.find_all("author"):
                name_elem = author.find("name")
                if name_elem:
                    authors.append(name_elem.text.strip())
            
            # Publication year and date
            # Extract date from published time (ISO format with T separator, e.g., 2024-01-15T10:30:00Z)
            published_elem = entry.find("published")
            year = None
            date = None
            if published_elem:
                try:
                    pub_date = published_elem.text.strip()
                    # Extract year
                    match = re.search(r"(\d{4})", pub_date)
                    if match:
                        year = int(match.group(1))
                    # Extract date in yyyy-mm-dd format
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", pub_date)
                    if date_match:
                        date = date_match.group(1)
                except ValueError:
                    pass
            
            # DOI and URL
            pdf_link = None
            abs_link = None
            link_doi = None
            for link in entry.find_all("link"):
                href = link.get("href", "")
                title_attr = link.get("title")
                if "/pdf/" in href or href.endswith(".pdf"):
                    pdf_link = href
                if "/abs/" in href:
                    abs_link = href
                if title_attr == "doi":
                    link_doi = href.replace("http://dx.doi.org/", "")

            id_elem = entry.find("id")
            arxiv_doi_elem = entry.find("arxiv:doi")
            doi = (arxiv_doi_elem.text.strip() if arxiv_doi_elem else None) or link_doi
            url = (id_elem.text.strip() if id_elem else None) or abs_link or pdf_link
            
            # Extract arxiv_id from URL
            arxiv_id = None
            if url:
                m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
                if m:
                    arxiv_id = m.group(1) + (m.group(2) or "")
            
            paper = {
                "title": title_text,
                "authors": authors,
                "abstract": abstract_text,
                "year": year,
                "doi": doi,
                "url": url,
                "arxiv_id": arxiv_id,
                "date": date,
            }
            papers.append(paper)
            
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {str(e)}")
    
    return papers


def fetch_arxiv_papers_debug_api(
    query: str, 
    max_results: int = 20, 
    sort: str = "relevance", 
    categories: list = None, 
    before: Optional[str] = None,
    after: Optional[str] = None
) -> list:
    """
    Search arXiv papers using the API.
    
    Args:
        query: Search query (supports ti:, abs:, au:, cat:, AND, OR)
        max_results: Maximum number of results
        sort: Sort order ("relevance" or "submittedDate")
        categories: Optional list of arXiv categories
        before: Optional date filter (YYYY-MM-DD format)
        after: Optional date filter (YYYY-MM-DD format)
        
    Returns:
        List of paper dictionaries
    """
    logger.info(f"Searching arXiv for: {query}")
    search_url = "http://export.arxiv.org/api/query"
    sort_param = "relevance" if sort == "relevance" else "submittedDate"
    
    cat_filter = ""
    if categories:
        cat_filter = " AND (" + " OR ".join([f"cat:{cat}" for cat in categories]) + ")"
    
    date_filter = ""
    try:
        start = "202401010000"
        end = "209912312359"
        
        if after:
            a = str(after)
            y = a[0:4]
            m = a[5:7] if len(a) >= 7 else "01"
            d = a[8:10] if len(a) >= 10 else "01"
            start = f"{y}{m}{d}0000"
        
        if before:
            b = str(before)
            y = b[0:4]
            m = b[5:7] if len(b) >= 7 else "12"
            d = b[8:10] if len(b) >= 10 else "31"
            end = f"{y}{m}{d}2359"
        
        if after or before:
            date_filter = f" AND submittedDate:[{start} TO {end}]"
    except Exception:
        pass
    
    q0 = str(query).strip()
    use_raw = any(k in q0 for k in ["ti:", "abs:", "au:", "cat:", "AND", "OR", "("])
    search_query_value = f"{q0}{cat_filter}{date_filter}" if use_raw else f"all:{q0}{cat_filter}{date_filter}"
    
    search_params = {
        "search_query": search_query_value,
        "max_results": max_results,
        "sortBy": sort_param,
        "sortOrder": "descending"
    }
    
    try:
        response = requests.get(search_url, params=search_params)
        logger.debug(f"ArXiv API URL: {response.url}")
        if response.status_code != 200:
            logger.error(f"arXiv search error: {response.status_code}")
            return []
        xml_data = response.text
        papers = parse_arxiv_xml(xml_data)
        logger.info(f"Found {len(papers)} papers from arXiv")
        return papers
    except Exception as e:
        logger.error(f"Error searching arXiv: {e}")
        return []


class ChoosePaperSignature(dspy.Signature):
    """
    You judge whether each paper is related work to the given idea.
    Input is a JSON array of {index,title,abstract} and the idea.
    Return a JSON array where each element has {index,is_related_work,reason}

    Task Definition:
    Determine if each paper in the input list qualifies as "related work" for the provided `basic_idea`.

    Definition of Related Work:
    A paper is considered related work if it meets ANY of the following criteria:
    1. Context Similarity: The paper focuses on a similar scenario or problem setting as the `basic_idea`.
       - Note: If a paper focuses on a specific domain (e.g., economics, ecology, medicine) that is NOT mentioned or implied in the `basic_idea`, it is likely NOT related work, even if some methods overlap.
    2. Core Concept/Method Overlap: The paper shares core concepts, entities, main tasks, implementation methods, or evaluation datasets with the `basic_idea`.
       - Note: Partial overlap is sufficient. The paper does NOT need to cover every aspect of the `basic_idea` or be identical.

    Evaluation Logic:
    - Be lenient with "relatedness". If there is a reasonable connection in task, method, or dataset, mark it as related.
    - Do not require the paper to solve exactly the same problem with exactly the same constraints.

    Examples:
    1. Basic Idea: "Using reinforcement learning for traffic signal control."
       - Paper: "Deep Q-Learning for optimizing traffic lights." -> Related (Method and task overlap).
       - Paper: "Traffic flow prediction using LSTM." -> Related (Same domain, similar goal).
       - Paper: "Reinforcement learning for stock market prediction." -> Not Related (Different domain/scenario).

    2. Basic Idea: "A new transformer architecture for long-document summarization."
       - Paper: "Efficient attention mechanisms for long sequences." -> Related (Method component overlap).
       - Paper: "Summarizing medical records with BERT." -> Related (Task overlap).
       - Paper: "Image classification with Vision Transformers." -> Not Related (Different modality/task, unless idea mentions cross-modal).
    """
    basic_idea = dspy.InputField(desc="The core basic idea text for context")
    papers = dspy.InputField(desc="JSON array of items: [{\"index\",\"title\",\"abstract\"}]")
    decisions = dspy.OutputField(desc="JSON array [{\"index\",\"is_related_work\",\"reason\"}]")


class ChoosePaper(dspy.Module):
    """Module for filtering papers based on relevance to the idea."""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        
        if config is None:
            ds_api_key = os.getenv("DS_API_KEY")
            if ds_api_key:
                config = {
                    "api_key": ds_api_key,
                    "api_base": os.getenv("DS_API_BASE_URL"),
                    "model": "openai/deepseek-v3.2"
                }
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    config = {
                        "api_key": openai_api_key,
                        "api_base": os.getenv("OPENAI_API_BASE_URL"),
                        "model": "openai/gpt-4o-mini"
                    }
                else:
                    raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY.")
        
        self.lm = dspy.LM(
            model=config.get("model", "gpt-4o-mini"),
            api_key=config["api_key"],
            api_base=config.get("api_base")
        )
        self.judge = dspy.ChainOfThought(ChoosePaperSignature)

    def forward(self, batch_sources: List[Dict[str, Any]], basic_idea: str) -> List[Dict[str, Any]]:
        """Judge if papers are related work."""
        papers_payload = []
        for idx, item in enumerate(batch_sources):
            p = item.get("p", {})
            papers_payload.append({
                "index": idx,
                "title": p.get("title", ""),
                "abstract": p.get("abstract", "")
            })
        
        with dspy.settings.context(lm=self.lm):
            out = self.judge(basic_idea=basic_idea, papers=json.dumps(papers_payload, ensure_ascii=False))
        
        raw = getattr(out, "decisions", "") or "[]"
        try:
            cleaned = str(raw).strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            decisions = json.loads(cleaned)
        except Exception:
            decisions = []
        
        norm = []
        for d in decisions:
            try:
                idx = int(d.get("index"))
                val = d.get("is_related_work")
                if isinstance(val, bool):
                    is_rel = val
                elif isinstance(val, str):
                    is_rel = val.strip().lower() in ("true", "1", "yes", "y")
                else:
                    is_rel = bool(val)
                reason = str(d.get("reason", "")).strip()
                norm.append({"index": idx, "is_related_work": is_rel, "reason": reason})
            except Exception:
                pass
        return norm


class PaperSearcher:
    """
    Searches for academic papers on arXiv with optional filtering.
    """
    
    def __init__(self, max_results_per_query: int = 8, enable_filtering: bool = True, batch_size: int = 8):
        """
        Initialize the paper searcher.
        
        Args:
            max_results_per_query: Maximum results per query
            enable_filtering: Whether to filter papers using LLM
            batch_size: Batch size for filtering
        """
        self.max_results = max_results_per_query
        self.enable_filtering = enable_filtering
        self.batch_size = batch_size
        logger.info(f"Initialized PaperSearcher with max_results={self.max_results}, enable_filtering={self.enable_filtering}")
    
    def search(self, queries: List[str], basic_idea: str = "", before: Optional[str] = None, after: Optional[str] = None) -> List[tuple]:
        """
        Search for papers using multiple queries with optional filtering.
        
        Args:
            queries: List of search queries
            basic_idea: Basic idea text for filtering (if enabled)
            before: Optional date filter (YYYY-MM-DD format)
            after: Optional date filter (YYYY-MM-DD format)
            
        Returns:
            List of tuples (Source, query_index) where query_index is the index of the query that found this source
        """
        all_sources_with_idx = []
        
        # Collect raw results with query indices
        raw_sources = []
        for q_idx, q in enumerate(queries):
            logger.info(f"Searching ArXiv for query {q_idx}: {q}")
            try:
                papers = fetch_arxiv_papers_debug_api(q, before=before, after=after, max_results=self.max_results) or []
                for p in papers:
                    raw_sources.append({"q": q, "q_idx": q_idx, "p": p})
            except Exception as e:
                logger.error(f"Error searching ArXiv for query '{q}': {e}")
        
        if not raw_sources:
            logger.info("No papers found")
            return []
        
        # Filter if enabled
        if self.enable_filtering and basic_idea:
            logger.info(f"Filtering {len(raw_sources)} papers using LLM...")
            chooser = ChoosePaper()
            final = []
            i = 0
            while i < len(raw_sources):
                batch = raw_sources[i:i+self.batch_size]
                decisions = chooser(batch, basic_idea)
                dec_map = {d.get("index"): d for d in decisions}
                allow = {idx: d for idx, d in dec_map.items() if d.get("is_related_work")}
                for j, item in enumerate(batch):
                    if j in allow:
                        final.append(item)
                i += self.batch_size
            raw_sources = final
            logger.info(f"Filtered to {len(raw_sources)} related papers")
        
        # Convert to Source objects and store query index
        for item in raw_sources:
            p = item.get("p", {})
            q_idx = item.get("q_idx", 0)
            try:
                source = self._convert_to_source(p)
                if source:
                    all_sources_with_idx.append((source, q_idx))
            except Exception as e:
                logger.error(f"Error converting paper to Source: {e}")
        
        # Deduplicate by title (keep first occurrence)
        seen_titles = set()
        unique_sources_with_idx = []
        for source, q_idx in all_sources_with_idx:
            normalized_title = ''.join(source.title.lower().split())
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_sources_with_idx.append((source, q_idx))
        
        logger.info(f"Found {len(unique_sources_with_idx)} unique papers")
        return unique_sources_with_idx
    
    def _convert_to_source(self, paper: dict) -> Optional[Source]:
        """Convert a paper dictionary to a Source object."""
        try:
            title = paper.get("title", "Untitled")
            url = paper.get("url", "")
            abstract = paper.get("abstract", "")
            authors = paper.get("authors", [])
            year = paper.get("year")
            date = paper.get("date")  # yyyy-mm-dd format
            doi = paper.get("doi")
            arxiv_id = paper.get("arxiv_id")
            
            # Build PDF URL from arxiv_id
            pdf_url = None
            if arxiv_id:
                pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Use date for timestamp if available, otherwise fall back to year
            timestamp = date if date else (str(year) if year else None)
            
            return Source(
                title=title,
                url=url,
                source_type=SourceType.PAPER,
                platform=Platform.ARXIV,
                description=abstract,
                authors=authors,
                year=year,
                metadata=paper,
                doi=doi,
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error converting paper to Source: {e}")
            return None
    
    def _deduplicate(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources based on normalized title."""
        seen = set()
        unique_sources = []
        for source in sources:
            normalized_title = ''.join(source.title.lower().split())
            if normalized_title not in seen:
                seen.add(normalized_title)
                unique_sources.append(source)
        return unique_sources

