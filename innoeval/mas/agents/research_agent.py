import asyncio
import inspect
import logging
import re
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from .extraction_agent import ExtractionAgent
from ..tools.searchers.models import Idea, SearchResults, SearchQuery, Source, SourceType
from ..tools.searchers.paper_searcher import PaperSearcher
from ..tools.searchers.web_searcher import WebSearcher
from ..tools.searchers.github_searcher_web import GithubWebSearcher
from ..tools.querygen.query_generator import QueryGenerator
from ..tools.querygen.reranker import rerank_articles_two_stage
from ..tools.querygen.readpage import read_page
from ..models.bge_singleton import get_bge_models
from ..tools.enricher import (
    enrich_papers_with_extraction,
    enrich_web_with_reports,
    enrich_code_with_rawtext,
    enrich_code_with_repo,
)

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Lightweight orchestrator: query -> search -> enrich(web/code) -> rerank -> enrich(paper) -> (optional) refine.
    Outputs SearchResults, with enrichment results and rerank scores written to Source.metadata.
    Note: First generate reports for web/github to enrich descriptions, then rerank.
    Paper extraction only applies to reranked results.

    Parallel optimization strategy:
    1. Search phase: paper/web/github searchers execute in parallel (using asyncio.gather)
    2. read_page phase: All URLs read in parallel (using asyncio.gather)
    3. Rerank phase: Sequential execution (PyTorch model is not thread-safe, avoid race conditions)
    4. Enrichment phase: web and code enrichment execute in parallel (calling LLM API, no shared state)

    Performance improvement: Expected 3-4x speedup (depending on network I/O and search API response time)
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ResearchAgent"
        self.temperature = config.get("temperature", 0.7)
        self.top_k = config.get("top_k", 10)
        self.enable_refine = config.get("enable_refine", False)

        self.max_results_per_query = config.get("max_results_per_query", 8)
        self.enable_paper_filtering = config.get("enable_paper_filtering", False)
        self.paper_batch_size = config.get("paper_batch_size", 8)
        self.web_max_results = config.get("web_max_results", 3)
        self.github_max_results = config.get("github_max_results", 3)
        self.get_future_paper = config.get("get_future_paper", True)

        # Save BGE model config, lazy load at execute time
        self.embedding_model_name = config.get("embedding_model_name", "BAAI/bge-base-en-v1.5")
        self.reranker_model_name = config.get("reranker_model_name", "BAAI/bge-reranker-base")
        self.hf_endpoint = config.get("hf_endpoint", "https://hf-mirror.com")
        self.embedding_model = None
        self.reranker_model = None

        self.query_generator = QueryGenerator()

        self.paper_searcher = PaperSearcher(
            max_results_per_query=self.max_results_per_query,
            enable_filtering=False,
            batch_size=self.paper_batch_size
        )
        self.web_searcher = WebSearcher(max_results_per_query=self.web_max_results)
        self.github_searcher = GithubWebSearcher(max_results_per_query=self.github_max_results)

        extraction_config = config.get(
            "extraction_config",
            {
                "name": "ExtractionAgent",
                "model_provider": config.get("model_provider", "default"),
                "extract_temperature": config.get("extract_temperature", 0.3),
                "_global_config": config.get("_global_config", {}),
            },
        )
        self.extraction_agent = ExtractionAgent(model, extraction_config)

    def _ensure_models_loaded(self):
        """Ensure BGE models are loaded (lazy load, called only at execute time)."""
        if self.embedding_model is None or self.reranker_model is None:
            frame = inspect.currentframe()
            logger.info(
                f"[{self.__class__.__name__}._ensure_models_loaded:{frame.f_lineno}] "
                f"Loading BGE models: {self.embedding_model_name} and {self.reranker_model_name}"
            )
            self.embedding_model, self.reranker_model = get_bge_models(
                embedding_model_name=self.embedding_model_name,
                reranker_model_name=self.reranker_model_name,
                hf_endpoint=self.hf_endpoint,
            )
            logger.info(f"[{self.__class__.__name__}._ensure_models_loaded:{frame.f_lineno}] BGE models loaded successfully")

    async def execute(self, idea: Idea, params: Optional[Dict[str, Any]] = None) -> SearchResults:
        # Lazy load BGE models (load only at execute time to avoid loading during initialization)
        self._ensure_models_loaded()
        
        params = params or {}
        depth = params.get("depth", 0) if self.enable_refine else 0
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}.execute:{frame.f_lineno}] Starting research_agent pipeline (depth={depth})")

        # 1) Generate initial queries
        initial_queries = self.query_generator.generate(idea)
        Q = initial_queries
        self._log_queries(Q)

        # Initialize accumulated sources
        all_papers: List[Source] = []
        all_web_pages: List[Source] = []
        all_github_repos: List[Source] = []

        idea_text = idea.get_full_text()

        # Maintain global deduplication set (based on URL)
        seen_paper_urls = set()
        seen_web_urls = set()
        seen_github_urls = set()
        
        # Maintain global enrichment set (based on URL), avoid duplicate enrichment
        enriched_web_urls = set()
        enriched_github_urls = set()

        # Use QueryGenerator to generate queries, no longer use hardcoded test logic
# Q.paper_queries = initial_queries.paper_queries  # Already assigned via Q = initial_queries
# Q.github_queries = initial_queries.github_queries  # Already assigned via Q = initial_queries
# Q.web_queries = initial_queries.web_queries  # Already assigned via Q = initial_queries

        # Iterative loop for depth times (when enable_refine=True and depth>0)
# If depth=0, execute one search without iteration
        iterations = depth if depth > 0 else 1
        for iteration in range(iterations):
            is_last_iteration = (iteration == iterations - 1)
            is_refine_iteration = (depth > 0 and not is_last_iteration)
            frame = inspect.currentframe()
            if depth > 0:
                logger.info(f"[{self.__class__.__name__}.execute:{frame.f_lineno}] Iteration {iteration + 1}/{depth}")
            else:
                logger.info(f"[{self.__class__.__name__}.execute:{frame.f_lineno}] Single search (no iteration)")

            # 2) Search (paper / web / github), use Q, with global deduplication and read_page
            new_papers, new_web_pages, new_github_repos = await self._run_search(
                idea, Q, params, seen_paper_urls, seen_web_urls, seen_github_urls
            )

            # 3) Merge new sources into historical sources
            all_papers.extend(new_papers)
            all_web_pages.extend(new_web_pages)
            all_github_repos.extend(new_github_repos)

            # 4) Rerank merged sources
            idea_full_text = idea.get_full_text()
            all_papers, all_web_pages, all_github_repos = await self._run_rerank(
                idea_full_text, all_papers, all_web_pages, all_github_repos
            )

            # 5) Enrich web and code (based on page_raw_text) to provide richer description for subsequent refine
            # Only enrich sources that haven't been enriched
            # ========== Parallel optimization: web and code enrichment execute in parallel ==========
            all_web_pages, all_github_repos = await asyncio.gather(
                self._run_enrich_web(idea_text, all_web_pages, params, enriched_web_urls),
                self._run_enrich_code(idea_text, all_github_repos, params, enriched_github_urls)
            )

            # 6) If not the last round and enable_refine=True, perform refine to generate new queries
            if is_refine_iteration and self.enable_refine:
                Q = self._run_refine(idea, all_papers, all_web_pages, all_github_repos, Q)
                self._log_refined_queries(Q)

        # 8) Enrich the final topk results (paper extraction and code repo analysis)
        all_papers, all_github_repos = await self._run_enrich_paper_and_code_final(
            idea_text, all_papers, all_github_repos, params
        )

        # 9) Return final results
        refined_queries = SearchQuery()
        if self.enable_refine and depth > 0:
            refined_queries = Q

        # 10) Get future papers (only executed when get_future_paper is True)
        if self.get_future_paper:
            future_papers = await self._get_future_papers(idea, initial_queries, params)
            all_papers = all_papers + future_papers

        return SearchResults(
            idea=idea,
            queries=initial_queries,  # Return initial queries
            papers=all_papers,
            github_repos=all_github_repos,
            web_pages=all_web_pages,
            kaggle_results=[],
            scholar_results=[],
            refined_queries=refined_queries,
        )


    async def _get_future_papers(self, idea: Idea, initial_queries: SearchQuery, params: Dict[str, Any]) -> List[Source]:
        """
        Get future papers (papers with timestamp >= before)
        
        Uses a separate search process, independent from regular search:
        - Uses independent seen_paper_urls set for deduplication
        - Uses initial_queries.paper_queries for search
        - Sets after=before to get papers from future time
        """
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Starting future papers search")
        
        # If paper_queries is empty, return empty list directly
        if not initial_queries.paper_queries:
            logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] No paper queries, skipping future papers search")
            return []
        
        # 1. Search using initial_queries.paper_queries
        seen_paper_urls = set()  # Independent deduplication set
        # Create SearchQuery object containing only paper_queries
        Q = SearchQuery(paper_queries=initial_queries.paper_queries)
        logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Starting search with {len(Q.paper_queries)} queries")
        self._log_queries(Q)
        
        all_papers: List[Source] = []
        before = params.get("before")
        # Consistent with execute: only enable multi-round refine when enable_refine=True
        depth = params.get("depth", 0) if self.enable_refine else 0
        
        if not before:
            logger.warning(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] No 'before' parameter provided, skipping future papers search")
            return []
        
        # Iterate loop depth times
        iterations = depth if depth > 0 else 1
        for iteration in range(iterations):
            is_last_iteration = (iteration == iterations - 1)
            is_refine_iteration = (depth > 0 and not is_last_iteration)
            frame = inspect.currentframe()
            if depth > 0:
                logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Iteration {iteration + 1}/{depth}")
            else:
                logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Single search (no iteration)")
            
            # Search: after=before means get papers with timestamp >= before
            paper_pairs = self.paper_searcher.search(
                Q.paper_queries, 
                basic_idea=idea.basic_idea or "", 
                before=None, 
                after=before
            )
            papers = [p for p, _ in paper_pairs if p.url and p.url not in seen_paper_urls]
            for p in papers:
                if p.url:
                    seen_paper_urls.add(p.url)
            all_papers.extend(papers)

            # Re-rank every round to avoid too many results in the last round
            idea_full_text = idea.get_full_text()
            all_papers, _, _ = await self._run_rerank(idea_full_text, all_papers, [], [])

            # Only refine on non-last rounds (and only when enable_refine is True), consistent with execute
            if is_refine_iteration and self.enable_refine:
                Q = self._run_refine(idea, all_papers, [], [], Q)
                self._log_refined_queries(Q)
        
        # 4. Enrich papers
        idea_text = idea.get_full_text()
        all_papers, _ = await self._run_enrich_paper_and_code_final(idea_text, all_papers, [], params)
        
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._get_future_papers:{frame.f_lineno}] Completed future papers search: found {len(all_papers)} papers (after={before})")
        self._log_sources("future_papers_search", all_papers)
        
        return all_papers

    async def _run_search(
        self,
        idea: Idea,
        queries: SearchQuery,
        params: Dict[str, Any],
        seen_paper_urls: set,
        seen_web_urls: set,
        seen_github_urls: set,
    ) -> (List[Source], List[Source], List[Source]):
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._run_search:{frame.f_lineno}] Starting parallel search")
        title = params.get("title")
        before = params.get("before")
        after = params.get("after")

        # ========== Parallel optimization: three searchers execute in parallel ==========
        async def search_papers():
            papers: List[Source] = []
            if queries.paper_queries:
                loop = asyncio.get_event_loop()
                paper_pairs = await loop.run_in_executor(
                    None,
                    self.paper_searcher.search,
                    queries.paper_queries,
                    idea.basic_idea or "",
                    before,
                    after
                )
                for src, q_idx in paper_pairs:
                    if src.metadata is None:
                        src.metadata = {}
                    src.metadata["query_index"] = q_idx
                    if q_idx is not None and q_idx < len(queries.paper_queries):
                        src.metadata["query"] = queries.paper_queries[q_idx]
                    papers.append(src)
            return papers

        async def search_web():
            web_pages: List[Source] = []
            if queries.web_queries:
                loop = asyncio.get_event_loop()
                web_pairs = await loop.run_in_executor(
                    None,
                    self.web_searcher.search,
                    queries.web_queries
                )
                for src, q_idx in web_pairs:
                    if src.metadata is None:
                        src.metadata = {}
                    src.metadata["query_index"] = q_idx
                    if q_idx is not None and q_idx < len(queries.web_queries):
                        src.metadata["query"] = queries.web_queries[q_idx]
                    web_pages.append(src)
            return web_pages

        async def search_github():
            github_repos: List[Source] = []
            if queries.github_queries:
                loop = asyncio.get_event_loop()
                repo_pairs = await loop.run_in_executor(
                    None,
                    self.github_searcher.search,
                    queries.github_queries
                )
                for src, q_idx in repo_pairs:
                    if src.metadata is None:
                        src.metadata = {}
                    src.metadata["query_index"] = q_idx
                    if q_idx is not None and q_idx < len(queries.github_queries):
                        src.metadata["query"] = queries.github_queries[q_idx]
                    github_repos.append(src)
            return github_repos

        # Execute three search tasks in parallel
        papers, web_pages, github_repos = await asyncio.gather(
            search_papers(),
            search_web(),
            search_github()
        )

        if title:
            papers = self._filter_by_title(papers, title)
            web_pages = self._filter_by_title(web_pages, title)
            github_repos = self._filter_by_title(github_repos, title)

        # Global deduplication: filter out results already seen in previous iterations
        papers = [p for p in papers if p.url and p.url not in seen_paper_urls]
        web_pages = [w for w in web_pages if w.url and w.url not in seen_web_urls]
        github_repos = [g for g in github_repos if g.url and g.url not in seen_github_urls]

        # Update seen sets
        for p in papers:
            if p.url:
                seen_paper_urls.add(p.url)
        for w in web_pages:
            if w.url:
                seen_web_urls.add(w.url)
        for g in github_repos:
            if g.url:
                seen_github_urls.add(g.url)

        # Execute read_page for web and github (after global deduplication)
        web_pages = await self._enrich_with_readpage(web_pages, "web")
        github_repos = await self._enrich_with_readpage(github_repos, "github")

        if title:
            papers = self._filter_by_title(papers, title)
            web_pages = self._filter_by_title(web_pages, title)
            github_repos = self._filter_by_title(github_repos, title)

        self._log_sources("paper_search", papers)
        self._log_sources("web_search", web_pages)
        self._log_sources("github_search", github_repos)

        return papers, web_pages, github_repos

    def _filter_by_title(self, sources: List[Source], original_title: str) -> List[Source]:
        """
        Filter out search results containing substrings of the original paper title.

        For papers: check title and description
        For web & code: check title, page_raw_text and description

        If any field contains a substring with length >= 0.8 * len(original_title), filter out that source.

        Args:
            sources: List of sources to filter
            original_title: Original paper title

        Returns:
            Filtered source list
        """
        if not original_title or not sources:
            return sources
        
        # Calculate minimum substring length
        min_substring_len = int(0.8 * len(original_title))
        if min_substring_len < 1:
            return sources
        
        # Generate all possible substrings (length >= min_substring_len)
        title_lower = original_title.lower()
        substrings = []
        for i in range(len(title_lower) - min_substring_len + 1):
            for j in range(i + min_substring_len, len(title_lower) + 1):
                substrings.append(title_lower[i:j])
        
        filtered_sources = []
        for source in sources:
            # Determine which fields to check based on source_type
            texts_to_check = []
            
            # All types check title and description
            if source.title:
                texts_to_check.append(source.title.lower())
            if source.description:
                texts_to_check.append(source.description.lower())
            
            # web and code types additionally check page_raw_text
            if source.source_type in [SourceType.WEBPAGE, SourceType.CODE]:
                if source.page_raw_text:
                    texts_to_check.append(source.page_raw_text.lower())
            
            # Check if any substring is contained
            should_filter = False
            for text in texts_to_check:
                for substring in substrings:
                    if substring in text:
                        should_filter = True
                        break
                if should_filter:
                    break
            
            # If no substring is contained, keep the source
            if not should_filter:
                filtered_sources.append(source)
        
        return filtered_sources

    async def _enrich_with_readpage(self, sources: List[Source], source_type: str) -> List[Source]:
        """
        Execute read_page on sources, filling page_raw_text and related metadata.
        Parallel optimization: read all URLs in parallel.

        Args:
            sources: List of Sources to enrich
            source_type: Source type ("web" or "github")

        Returns:
            Enriched Source list
        """
        if not sources:
            return sources

        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._enrich_with_readpage:{frame.f_lineno}] Starting parallel read_page for {source_type} (count={len(sources)})"
        )

        async def read_single_page(src: Source) -> Source:
            """Read single page in parallel"""
            if not src.url:
                return src

            try:
                loop = asyncio.get_event_loop()
                page_data = await loop.run_in_executor(None, read_page, src.url)

                if isinstance(page_data, dict):
                    # Extract raw text from the dict returned by read_page
                    src.page_raw_text = page_data.get("raw", "")
                    # Also store metadata if available
                    if "md" in page_data and page_data["md"]:
                        md = page_data["md"]
                        if isinstance(md, dict):
                            src.page_title = md.get("title")
                            src.page_headings = md.get("headings", [])
                            src.page_links = md.get("links", [])
                else:
                    src.page_raw_text = str(page_data) if page_data else ""
            except Exception as e:
                logger.warning(f"Failed to read page for {source_type} {src.url}: {e}")
                src.page_raw_text = ""

            return src

        # Read all pages in parallel
        enriched_sources = await asyncio.gather(
            *[read_single_page(src) for src in sources]
        )

        logger.info(
            f"[{self.__class__.__name__}._enrich_with_readpage:{frame.f_lineno}] Completed parallel read_page for {source_type}"
        )
        return list(enriched_sources)

    async def _run_enrich_web(
        self,
        idea_text: str,
        web_pages: List[Source],
        params: Dict[str, Any],
        enriched_web_urls: set,
    ):
        frame = inspect.currentframe()
        # Filter out unenriched web_pages
        to_enrich = [wp for wp in web_pages if wp.url and wp.url not in enriched_web_urls]
        if not to_enrich:
            logger.info(
                f"[{self.__class__.__name__}._run_enrich_web:{frame.f_lineno}] All web pages already enriched (total={len(web_pages)})"
            )
            return web_pages
        
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_web:{frame.f_lineno}] Starting enrich web (total={len(web_pages)}, to_enrich={len(to_enrich)})"
        )
        web_temp = params.get("web_temperature", self.temperature)

        # Only enrich web_pages that are not yet enriched
        enriched_pages = await enrich_web_with_reports(self._call_model, idea_text, to_enrich, web_temp)
        self._log_enrich("web_enrich", enriched_pages, key="web_report")
        
        # Update enriched_web_urls set
        for wp in enriched_pages:
            if wp.url:
                enriched_web_urls.add(wp.url)

        return web_pages

    async def _run_enrich_code(
        self,
        idea_text: str,
        github_repos: List[Source],
        params: Dict[str, Any],
        enriched_github_urls: set,
    ):
        frame = inspect.currentframe()
        # Filter out unenriched github_repos
        to_enrich = [gr for gr in github_repos if gr.url and gr.url not in enriched_github_urls]
        if not to_enrich:
            logger.info(
                f"[{self.__class__.__name__}._run_enrich_code:{frame.f_lineno}] All github repos already enriched (total={len(github_repos)})"
            )
            return github_repos
        
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_code:{frame.f_lineno}] Starting enrich code (total={len(github_repos)}, to_enrich={len(to_enrich)})"
        )
        code_temp = params.get("code_temperature", self.temperature)

        # Only enrich github_repos that are not yet enriched
        enriched_repos = await enrich_code_with_rawtext(self._call_model, idea_text, to_enrich, code_temp)
        self._log_enrich("code_enrich", enriched_repos, key="code_report")
        
        # Update enriched_github_urls set
        for gr in enriched_repos:
            if gr.url:
                enriched_github_urls.add(gr.url)

        return github_repos

    async def _run_enrich_paper_and_code_final(
        self,
        idea_text: str,
        papers: List[Source],
        github_repos: List[Source],
        params: Dict[str, Any],
    ):
        """
        Enrich the final topk results:
        - papers: Use extraction_agent to extract PDF
        - github_repos: Use enrich_code_with_repo to generate reports based on repo context
        """
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._run_enrich_paper_and_code_final:{frame.f_lineno}] Starting final enrich paper/code (papers={len(papers)}, github={len(github_repos)})"
        )
        code_temp = params.get("code_temperature", self.temperature)

        # Paper extraction
        papers = await enrich_papers_with_extraction(papers, self.extraction_agent)
        self._log_enrich("paper_enrich_final", papers, key="paper_extract")

        # Code reports (based on repo_context/readme)
        github_repos = await enrich_code_with_repo(self._call_model, idea_text, github_repos, code_temp)
        self._log_enrich("code_enrich_final", github_repos, key="code_report")

        return papers, github_repos

    async def _run_rerank(
        self,
        idea_full_text: str,
        papers: List[Source],
        web_pages: List[Source],
        github_repos: List[Source],
    ):
        frame = inspect.currentframe()
        logger.info(f"[{self.__class__.__name__}._run_rerank:{frame.f_lineno}] Starting rerank with idea_full_text")
        papers_ranked = self._rerank_single("papers", idea_full_text, papers)
        web_ranked = self._rerank_single("web", idea_full_text, web_pages)
        github_ranked = self._rerank_single("github", idea_full_text, github_repos)
        return papers_ranked, web_ranked, github_ranked

    def _rerank_single(self, label: str, idea_full_text: str, sources: List[Source]) -> List[Source]:
        frame = inspect.currentframe()
        if not sources:
            logger.info(f"[{self.__class__.__name__}._rerank_single:{frame.f_lineno}] {label}: no sources to rerank")
            return []

        items = []
        for idx, src in enumerate(sources):
            # Extract article by label: paper is description, web and code are page_raw_text
            if label == "papers":
                text = src.description or src.title
            elif label in ["web", "github"]:
                text = src.description or src.page_raw_text or src.title
            else:
                text = self._source_text(src)
                if not text:
                    text = src.description or src.title
            items.append((idx, src, f"[idx:{idx}] {text}"))

        article_list = [item[2] for item in items]
        
        try:
            reranked = rerank_articles_two_stage(
                core_article=idea_full_text,
                article_list=article_list,
                top_k=self.top_k,
                source_type=label,
                enable_llm_scoring=True,
                embedding_model=self.embedding_model,
                reranker_model=self.reranker_model,
            )
        except Exception as e:
            logger.warning(f"[{self.__class__.__name__}._rerank_single:{frame.f_lineno}] Rerank failed: {e}, using fallback ranking")
            # Fallback logic: use simple text similarity ranking to ensure only top_k resources are returned
            return self._fallback_rerank(idea_full_text, sources, label)

        ranked_sources: List[Source] = []
        for content, emb_score, rerank_score in reranked:
            idx = self._extract_idx(content)
            if idx is None or idx >= len(items):
                continue
            src = items[idx][1]
            if src.metadata is None:
                src.metadata = {}
            src.metadata["rerank"] = {
                "embedding_score": emb_score,
                "reranker_score": rerank_score,
            }
            ranked_sources.append(src)

        self._log_rerank(label, ranked_sources)
        return ranked_sources

    def _run_refine(
        self,
        idea: Idea,
        papers: List[Source],
        web_pages: List[Source],
        github_repos: List[Source],
        queries: SearchQuery,
    ) -> SearchQuery:
        refined = SearchQuery()

        idea_full_text = idea.get_full_text()
        paper_scored = self._collect_scored_sources(papers)
        if paper_scored and queries.paper_queries:
            top_sources = [s for _, s in paper_scored]
            similarity_scores = [float(score) for score, _ in paper_scored]
            source_queries = [s.metadata.get("query", "") if s.metadata else "" for s in top_sources]
            refined.paper_queries = self.query_generator.refine_paper_queries(
                idea_full_text=idea_full_text,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=queries.paper_queries,
            )

        web_scored = self._collect_scored_sources(web_pages)
        if web_scored and queries.web_queries:
            top_sources = [s for _, s in web_scored]
            similarity_scores = [float(score) for score, _ in web_scored]
            source_queries = [s.metadata.get("query", "") if s.metadata else "" for s in top_sources]
            refined.web_queries = self.query_generator.refine_web_queries(
                idea_full_text=idea_full_text,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=queries.web_queries,
            )

        github_scored = self._collect_scored_sources(github_repos)
        if github_scored and queries.github_queries:
            top_sources = [s for _, s in github_scored]
            similarity_scores = [float(score) for score, _ in github_scored]
            source_queries = [s.metadata.get("query", "") if s.metadata else "" for s in top_sources]
            refined.github_queries = self.query_generator.refine_github_queries(
                idea_full_text=idea_full_text,
                top_sources=top_sources,
                similarity_scores=similarity_scores,
                source_queries=source_queries,
                original_queries=queries.github_queries,
            )

        return refined

    def _log_queries(self, queries: SearchQuery) -> None:
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._log_queries:{frame.f_lineno}] Generated queries | paper=%d web=%d github=%d scholar=%d kaggle=%d",
            len(queries.paper_queries),
            len(queries.web_queries),
            len(queries.github_queries),
            len(queries.scholar_queries),
            len(queries.kaggle_queries),
        )
        logger.info(
            f"[{self.__class__.__name__}._log_queries:{frame.f_lineno}] Queries detail | paper=%s | web=%s | github=%s",
            queries.paper_queries,
            queries.web_queries,
            queries.github_queries,
        )

    def _log_sources(self, stage: str, sources: List[Source], limit: int = 5) -> None:
        frame = inspect.currentframe()
        if not sources:
            logger.info(f"[{self.__class__.__name__}._log_sources:{frame.f_lineno}] %s: 0 results", stage)
            return
        samples = []
        for idx, src in enumerate(sources[:limit]):
            q = src.metadata.get("query") if src.metadata else None
            samples.append(f"{idx+1}. {src.title or 'N/A'} (query={q})")
        logger.info(f"[{self.__class__.__name__}._log_sources:{frame.f_lineno}] %s: %d results. Top%d: %s", stage, len(sources), limit, " | ".join(samples))

    def _log_enrich(self, stage: str, sources: List[Source], key: str, limit: int = 3) -> None:
        frame = inspect.currentframe()
        total = len(sources)
        with_key = sum(1 for s in sources if s.metadata and key in s.metadata)
        logger.info(f"[{self.__class__.__name__}._log_enrich:{frame.f_lineno}] %s: %d items, %d with %s", stage, total, with_key, key)
        if with_key == 0:
            return
        samples = []
        for src in sources:
            if not src.metadata or key not in src.metadata:
                continue
            content = src.metadata.get(key)
            text = ""
            if isinstance(content, dict):
                text = content.get("summary") or content.get("report_content") or str(content)
            else:
                text = str(content)
            text = (text[:180] + "...") if text and len(text) > 200 else text
            samples.append(f"{src.title or 'N/A'} -> {text}")
            if len(samples) >= limit:
                break
        if samples:
            logger.info(f"[{self.__class__.__name__}._log_enrich:{frame.f_lineno}] %s samples: %s", stage, " | ".join(samples))

    def _log_rerank(self, label: str, sources: List[Source], limit: int = 5) -> None:
        frame = inspect.currentframe()
        if not sources:
            logger.info(f"[{self.__class__.__name__}._log_rerank:{frame.f_lineno}] rerank[%s]: 0 items", label)
            return
        top = []
        for src in sources[:limit]:
            rerank_meta = src.metadata.get("rerank") if src.metadata else {}
            score = None
            if isinstance(rerank_meta, dict):
                score = rerank_meta.get("reranker_score")
            top.append(f"{src.title or 'N/A'} (score={score})")
        logger.info(f"[{self.__class__.__name__}._log_rerank:{frame.f_lineno}] rerank[%s]: total=%d, top%d=%s", label, len(sources), limit, " | ".join(top))

    def _log_refined_queries(self, refined: SearchQuery) -> None:
        frame = inspect.currentframe()
        has_any = refined.paper_queries or refined.web_queries or refined.github_queries
        if not has_any:
            logger.info(f"[{self.__class__.__name__}._log_refined_queries:{frame.f_lineno}] Refine: no new queries")
            return
        logger.info(
            f"[{self.__class__.__name__}._log_refined_queries:{frame.f_lineno}] Refine queries | paper=%s | web=%s | github=%s",
            refined.paper_queries,
            refined.web_queries,
            refined.github_queries,
        )

    def _collect_scored_sources(self, sources: List[Source]) -> List[tuple]:
        scored = []
        for src in sources:
            rerank_meta = src.metadata.get("rerank") if src.metadata else None
            score = None
            if isinstance(rerank_meta, dict):
                score = rerank_meta.get("reranker_score")
            if score is not None:
                scored.append((float(score), src))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: self.top_k]

    def _source_text(self, source: Source) -> str:
        meta = source.metadata or {}
        if "paper_extract" in meta:
            extract = meta["paper_extract"]
            if isinstance(extract, dict):
                parts = []
                for k, v in extract.items():
                    if isinstance(v, str):
                        parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        parts.append(f"{k}: {' '.join([str(i) for i in v])}")
                if parts:
                    return "\n".join(parts)
            return str(extract)
        if "web_report" in meta:
            rep = meta["web_report"]
            if isinstance(rep, dict):
                return rep.get("report_content") or rep.get("summary") or str(rep)
            return str(rep)
        if "code_report" in meta:
            rep = meta["code_report"]
            if isinstance(rep, dict):
                return rep.get("report_content") or rep.get("summary") or str(rep)
            return str(rep)
        if source.page_raw_text:
            return source.page_raw_text
        if source.repo_context:
            return source.repo_context
        return source.description or ""

    def _fallback_rerank(self, idea_full_text: str, sources: List[Source], label: str) -> List[Source]:
        """
        Fallback rerank logic: used when BGE model fails to load or rerank fails.
        Uses simple keyword matching to calculate similarity, ensuring only top_k resources are returned.
        
        Args:
            idea_full_text: Full text of the research idea
            sources: List of sources to rerank
            label: Source type label ("papers", "web", "github")
        
        Returns:
            Reranked source list (up to top_k)
        """
        if not sources:
            return []
        
        # Extract keywords from idea (simple word extraction, remove common stopwords)
        idea_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', idea_full_text.lower()))
        # Simple stopword filtering
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'way', 'use', 'her', 'she', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'way', 'use'}
        idea_words = idea_words - stop_words
        
        # Calculate similarity score for each source
        scored_sources = []
        for src in sources:
            # Extract source text (consistent with _rerank_single)
            if label == "papers":
                text = src.description or src.title or ""
            elif label in ["web", "github"]:
                text = src.description or src.page_raw_text or src.title or ""
            else:
                text = self._source_text(src) or src.description or src.title or ""
            
            text_lower = text.lower()
            # Count matched keywords
            matched_words = sum(1 for word in idea_words if word in text_lower)
            # Calculate similarity score: matched keyword count / total idea keywords (avoid division by zero)
            score = matched_words / len(idea_words) if idea_words else 0.0
            
            scored_sources.append((score, src))
        
        # Sort by score in descending order
        scored_sources.sort(key=lambda x: x[0], reverse=True)
        
        # Only return top_k resources
        result = [src for _, src in scored_sources[:self.top_k]]
        
        # Add metadata to results (mark as fallback rerank)
        for src in result:
            if src.metadata is None:
                src.metadata = {}
            if "rerank" not in src.metadata:
                src.metadata["rerank"] = {}
            src.metadata["rerank"]["fallback"] = True
        
        frame = inspect.currentframe()
        logger.info(
            f"[{self.__class__.__name__}._fallback_rerank:{frame.f_lineno}] "
            f"Fallback rerank completed for {label}: {len(result)}/{len(sources)} sources selected"
        )
        
        return result

    @staticmethod
    def _extract_idx(text: str) -> Optional[int]:
        if not text.startswith("[idx:"):
            return None
        try:
            prefix = text.split("]", 1)[0]
            idx_str = prefix.replace("[idx:", "")
            return int(idx_str)
        except Exception:
            return None


__all__ = ["ResearchAgent"]

