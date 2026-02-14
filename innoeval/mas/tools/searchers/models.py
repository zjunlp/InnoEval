"""
Data Models for Search Agent

This module defines the core data structures used by the Search Agent:
- Idea: Input structure representing a research idea
- SearchQuery: Structured queries for different platforms
- Source: Individual search result (paper, repo, webpage)
- SearchResults: Aggregated results from all platforms
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid
from enum import Enum


class SourceType(Enum):
    """Type of source"""
    PAPER = "paper"
    CODE = "code"
    WEBPAGE = "webpage"


class Platform(Enum):
    """Search platform"""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    GITHUB = "github"
    GOOGLE_SEARCH = "google_search"
    GOOGLE_SCHOLAR = "google_scholar"
    KAGGLE = "kaggle"


@dataclass
class Idea:
    """
    Represents a research idea with all its components.

    Attributes:
        basic_idea: A summary of the core concept
        motivation: The motivation behind the research
        research_question: The main research question
        method: Proposed methodology
        experimental_setting: Experimental setup description
        expected_results: Expected outcomes (optional)
        <field>_list: List of atomic claims for <field>
    """
    basic_idea: str
    motivation: str
    research_question: str
    method: str
    experimental_setting: Optional[str] = None
    expected_results: Optional[str] = None
    basic_idea_list: List[str] = field(default_factory=list)
    motivation_list: List[str] = field(default_factory=list)
    research_question_list: List[str] = field(default_factory=list)
    method_list: List[str] = field(default_factory=list)
    experimental_setting_list: Optional[List[str]] = None
    expected_results_list: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "basic_idea": self.basic_idea,
            "motivation": self.motivation,
            "research_question": self.research_question,
            "method": self.method,
            "experimental_setting": self.experimental_setting,
            "expected_results": self.expected_results,
            "basic_idea_list": self.basic_idea_list,
            "motivation_list": self.motivation_list,
            "research_question_list": self.research_question_list,
            "method_list": self.method_list,
            "experimental_setting_list": self.experimental_setting_list,
            "expected_results_list": self.expected_results_list,
        }

    def get_full_text(self, part: Optional[List[str]] = None) -> str:
        """
        Get full text representation of the idea.
        
        Args:
            part: Optional list of field names to include. Valid values:
                  'basic_idea', 'motivation', 'research_question', 'method', 
                  'experimental_setting', 'expected_results'.
                  If None, includes all fields.
        
        Returns:
            Formatted text string with selected fields in fixed order
        """
        # Define all valid parts in the specified order
        valid_parts_order = [
            'basic_idea', 'motivation', 'research_question', 
            'method', 'experimental_setting', 'expected_results'
        ]
        
        # Field mapping: (label, content)
        field_map = {
            'basic_idea': ('Basic Idea', self.basic_idea),
            'motivation': ('Motivation', self.motivation),
            'research_question': ('Research Question', self.research_question),
            'method': ('Method', self.method),
            'experimental_setting': ('Experimental Setting', self.experimental_setting),
            'expected_results': ('Expected Results', self.expected_results),
        }
        
        # Determine which fields to include
        fields_to_include = set(part) if part else set(valid_parts_order)
        
        # Always iterate in the fixed order, but only include fields that are requested
        parts = []
        for p in valid_parts_order:
            if p in fields_to_include and p in field_map:
                label, content = field_map[p]
                if content:  # Only add if content is not empty
                    parts.append(f"### {label}\n{content}")

        return "\n\n".join(parts) if parts else ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Idea":
        return cls(
            basic_idea=data.get("basic_idea", ""),
            motivation=data.get("motivation", ""),
            research_question=data.get("research_question", ""),
            method=data.get("method", ""),
            experimental_setting=data.get("experimental_setting", ""),
            expected_results=data.get("expected_results"),
            basic_idea_list=data.get("basic_idea_list", []),
            motivation_list=data.get("motivation_list", []),
            research_question_list=data.get("research_question_list", []),
            method_list=data.get("method_list", []),
            experimental_setting_list=data.get("experimental_setting_list", []),
            expected_results_list=data.get("expected_results_list", []),
        )
    
    @classmethod
    def from_lists(
        cls,
        basic_idea_list: List[str],
        motivation_list: List[str],
        research_question_list: List[str],
        method_list: List[str],
        experimental_setting_list: Optional[List[str]] = None,
        expected_results_list: Optional[List[str]] = None,
    ) -> "Idea":
        def _join_list(items: Optional[List[str]]) -> Optional[str]:
            """Join list items with '\n', return None if items is None or empty"""
            if not items:
                return None
            result = "\n".join(str(item) for item in items if item)
            return result if result else None

        return cls(
            basic_idea=_join_list(basic_idea_list),
            motivation=_join_list(motivation_list),
            research_question=_join_list(research_question_list),
            method=_join_list(method_list),
            experimental_setting=_join_list(experimental_setting_list),
            expected_results=_join_list(expected_results_list),
            basic_idea_list=basic_idea_list,
            motivation_list=motivation_list,
            research_question_list=research_question_list,
            method_list=method_list,
            experimental_setting_list=experimental_setting_list,
            expected_results_list=expected_results_list,
        )


@dataclass
class SearchQuery:
    """
    Represents structured queries for different platforms.

    Attributes:
        paper_queries: Queries for academic paper search (arXiv, Semantic Scholar, PubMed)
        code_queries: Queries for code repository search (GitHub)
        web_queries: Queries for web search (Google)
        scholar_queries: Queries for Google Scholar
    """
    paper_queries: List[str] = field(default_factory=list)
    github_queries: List[str] = field(default_factory=list)
    kaggle_queries: List[str] = field(default_factory=list)
    web_queries: List[str] = field(default_factory=list)
    scholar_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary"""
        return {
            "paper_queries": self.paper_queries,
            "github_queries": self.github_queries,
            "kaggle_queries": self.kaggle_queries,
            "web_queries": self.web_queries,
            "scholar_queries": self.scholar_queries
        }

    def get_all_queries(self) -> List[str]:
        """Get all queries as a flat list"""
        return (self.paper_queries + self.github_queries + self.kaggle_queries +
                self.web_queries + self.scholar_queries)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchQuery":
        return cls(
            paper_queries=data.get("paper_queries", []),
            github_queries=data.get("github_queries", []),
            kaggle_queries=data.get("kaggle_queries", []),
            web_queries=data.get("web_queries", []),
            scholar_queries=data.get("scholar_queries", []),
        )


@dataclass
class Source:
    """
    Represents a single search result source.

    Attributes:
        title: Title of the source
        url: URL or link to the source
        source_type: Type of source (paper, code, webpage)
        platform: Platform where the source was found
        description: Brief description or abstract
        authors: List of authors (for papers)
        year: Publication year (for papers)
        citations: Citation count (for papers)
        metadata: Additional metadata
    """
    title: str
    url: str
    source_type: SourceType
    platform: Platform
    description: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    citations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    normalized_title: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    timestamp: Optional[str] = None
    kaggle_ref: Optional[str] = None
    kaggle_item_type: Optional[str] = None
    kaggle_subtitle: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_tldr: Optional[str] = None
    s2_open_access_pdf: Optional[str] = None
    web_source_name: Optional[str] = None
    web_rank: Optional[int] = None
    scholar_publication_info: Optional[str] = None
    scholar_rank: Optional[int] = None
    scholar_cited_by: Optional[int] = None
    page_title: Optional[str] = None
    page_headings: Optional[List[str]] = None
    page_links: Optional[List[str]] = None
    page_raw_text: Optional[str] = None
    repo_context: Optional[str] = None
    repo_readme: Optional[str] = None

    def __post_init__(self):
        if self.title and not self.normalized_title:
            self.normalized_title = ''.join(self.title.lower().split())
        if not self.id:
            base = f"{self.platform.value}:{self.url or ''}:{self.normalized_title or self.title}"
            self.id = str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type.value,
            "platform": self.platform.value,
            "description": self.description,
            "authors": self.authors,
            "year": self.year,
            "citations": self.citations,
            "metadata": self.metadata,
            "id": self.id,
            "normalized_title": self.normalized_title,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "timestamp": self.timestamp,
            "kaggle_ref": self.kaggle_ref,
            "kaggle_item_type": self.kaggle_item_type,
            "kaggle_subtitle": self.kaggle_subtitle,
            "arxiv_id": self.arxiv_id,
            "s2_tldr": self.s2_tldr,
            "s2_open_access_pdf": self.s2_open_access_pdf,
            "web_source_name": self.web_source_name,
            "web_rank": self.web_rank,
            "scholar_publication_info": self.scholar_publication_info,
            "scholar_rank": self.scholar_rank,
            "scholar_cited_by": self.scholar_cited_by,
            "page_title": self.page_title,
            "page_headings": self.page_headings,
            "page_links": self.page_links,
            "page_raw_text": self.page_raw_text,
            "repo_context": self.repo_context,
            "repo_readme": self.repo_readme,
        }

    def __str__(self) -> str:
        """String representation"""
        parts = [f"[{self.source_type.value.upper()}] {self.title}"]

        if self.authors:
            authors_str = ", ".join(self.authors[:3])
            if len(self.authors) > 3:
                authors_str += " et al."
            parts.append(f"Authors: {authors_str}")

        if self.year:
            parts.append(f"Year: {self.year}")

        if self.citations:
            parts.append(f"Citations: {self.citations}")

        parts.append(f"Platform: {self.platform.value}")
        parts.append(f"URL: {self.url}")

        if self.description:
            desc = self.description[:200] + "..." if len(self.description) > 200 else self.description
            parts.append(f"Description: {desc}")

        return "\n".join(parts)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            source_type=SourceType(data.get("source_type", SourceType.WEBPAGE.value)),
            platform=Platform(data.get("platform", Platform.GOOGLE_SEARCH.value)),
            description=data.get("description"),
            authors=data.get("authors"),
            year=data.get("year"),
            citations=data.get("citations"),
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            normalized_title=data.get("normalized_title"),
            doi=data.get("doi"),
            pdf_url=data.get("pdf_url"),
            timestamp=data.get("timestamp"),
            kaggle_ref=data.get("kaggle_ref"),
            kaggle_item_type=data.get("kaggle_item_type"),
            kaggle_subtitle=data.get("kaggle_subtitle"),
            arxiv_id=data.get("arxiv_id"),
            s2_tldr=data.get("s2_tldr"),
            s2_open_access_pdf=data.get("s2_open_access_pdf"),
            web_source_name=data.get("web_source_name"),
            web_rank=data.get("web_rank"),
            scholar_publication_info=data.get("scholar_publication_info"),
            scholar_rank=data.get("scholar_rank"),
            scholar_cited_by=data.get("scholar_cited_by"),
            page_title=data.get("page_title"),
            page_headings=data.get("page_headings"),
            page_links=data.get("page_links"),
            page_raw_text=data.get("page_raw_text"),
            repo_context=data.get("repo_context"),
            repo_readme=data.get("repo_readme"),
        )


@dataclass
class SearchResults:
    """
    Aggregated search results from all platforms.

    Attributes:
        idea: The original idea that was searched
        queries: The queries that were generated
        papers: List of paper sources
        github_repos: List of GitHub repository sources
        kaggle_results: List of Kaggle datasets/notebooks
        web_pages: List of web page sources
        scholar_results: List of Google Scholar results
        refined_queries: Refined queries generated after iteration (optional)
        total_count: Total number of sources found
    """
    idea: Idea
    queries: SearchQuery
    papers: List[Source] = field(default_factory=list)
    github_repos: List[Source] = field(default_factory=list)
    kaggle_results: List[Source] = field(default_factory=list)
    web_pages: List[Source] = field(default_factory=list)
    scholar_results: List[Source] = field(default_factory=list)
    refined_queries: Optional[SearchQuery] = None

    @property
    def total_count(self) -> int:
        """Get total number of sources"""
        return (len(self.papers) + len(self.github_repos) + len(self.kaggle_results) +
                len(self.web_pages) + len(self.scholar_results))

    def get_all_sources(self) -> List[Source]:
        """Get all sources as a flat list"""
        return (self.papers + self.github_repos + self.kaggle_results +
                self.web_pages + self.scholar_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "idea": self.idea.to_dict(),
            "queries": self.queries.to_dict(),
            "papers": [s.to_dict() for s in self.papers],
            "github_repos": [s.to_dict() for s in self.github_repos],
            "kaggle_results": [s.to_dict() for s in self.kaggle_results],
            "web_pages": [s.to_dict() for s in self.web_pages],
            "scholar_results": [s.to_dict() for s in self.scholar_results],
            "total_count": self.total_count,
        }
        if self.refined_queries is not None:
            result["refined_queries"] = self.refined_queries.to_dict()
        return result

    def summary(self) -> str:
        """Get a summary of the search results"""
        lines = [
            "=" * 80,
            "SEARCH RESULTS SUMMARY",
            "=" * 80,
            f"Total Sources Found: {self.total_count}",
            f"  - Papers: {len(self.papers)}",
            f"  - GitHub Repositories: {len(self.github_repos)}",
            f"  - Kaggle Results: {len(self.kaggle_results)}",
            f"  - Web Pages: {len(self.web_pages)}",
            f"  - Scholar Results: {len(self.scholar_results)}",
            "=" * 80
        ]
        return "\n".join(lines)

    def detailed_report(self, max_sources_per_type: int = 5) -> str:
        """Generate a detailed report of the search results"""
        lines = [self.summary(), ""]

        # Papers
        if self.papers:
            lines.append("\n" + "=" * 80)
            lines.append("PAPERS")
            lines.append("=" * 80)
            for i, paper in enumerate(self.papers[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {paper}")
                lines.append("-" * 80)

        # GitHub Repositories
        if self.github_repos:
            lines.append("\n" + "=" * 80)
            lines.append("GITHUB REPOSITORIES")
            lines.append("=" * 80)
            for i, repo in enumerate(self.github_repos[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {repo}")
                lines.append("-" * 80)

        # Kaggle Results
        if self.kaggle_results:
            lines.append("\n" + "=" * 80)
            lines.append("KAGGLE RESULTS")
            lines.append("=" * 80)
            for i, item in enumerate(self.kaggle_results[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {item}")
                lines.append("-" * 80)

        # Web Pages
        if self.web_pages:
            lines.append("\n" + "=" * 80)
            lines.append("WEB PAGES")
            lines.append("=" * 80)
            for i, page in enumerate(self.web_pages[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {page}")
                lines.append("-" * 80)

        # Scholar Results
        if self.scholar_results:
            lines.append("\n" + "=" * 80)
            lines.append("SCHOLAR RESULTS")
            lines.append("=" * 80)
            for i, result in enumerate(self.scholar_results[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {result}")
                lines.append("-" * 80)

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResults":
        refined_queries = None
        if "refined_queries" in data and data["refined_queries"]:
            refined_queries = SearchQuery.from_dict(data["refined_queries"])
        return cls(
            idea=Idea.from_dict(data.get("idea", {})),
            queries=SearchQuery.from_dict(data.get("queries", {})),
            papers=[Source.from_dict(s) for s in data.get("papers", [])],
            github_repos=[Source.from_dict(s) for s in data.get("github_repos", [])],
            kaggle_results=[Source.from_dict(s) for s in data.get("kaggle_results", [])],
            web_pages=[Source.from_dict(s) for s in data.get("web_pages", [])],
            scholar_results=[Source.from_dict(s) for s in data.get("scholar_results", [])],
            refined_queries=refined_queries,
        )
