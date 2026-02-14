"""
Utilities for extracting GitHub repositories from papers.
"""

import logging
from typing import List, Optional

from .models import Source, SourceType, Platform
from ..querygen.readpage import read_page

logger = logging.getLogger(__name__)


def normalize_github_url(url: str) -> Optional[str]:
    """
    Normalize GitHub URL by:
    1. Remove .git suffix if present
    2. Extract github.com/owner/repo format (exactly two fields after github.com/)
    3. Remove trailing slashes
    
    Args:
        url: Raw GitHub URL
        
    Returns:
        Normalized URL or None if invalid
    """
    if not url or "github.com" not in url:
        return None
    
    # Remove .git suffix if present
    if ".git" in url:
        url = url.split(".git")[0]
    
    # Find github.com position
    github_idx = url.find("github.com/")
    if github_idx == -1:
        return None
    
    # Extract part after github.com/
    after_github = url[github_idx + len("github.com/"):]
    
    # Split by '/' and take first two fields
    parts = [p for p in after_github.split("/") if p]
    
    if len(parts) < 2:
        return None
    
    # Take only owner and repo (first two fields)
    owner = parts[0]
    repo = parts[1]
    
    # Remove trailing slashes and return normalized URL
    normalized = f"https://github.com/{owner}/{repo}".rstrip("/")
    return normalized


def compute_ngram_overlap(text1: str, text2: str, n: int) -> float:
    """
    Compute n-gram overlap between two texts.
    
    Args:
        text1: First text
        text2: Second text
        n: N-gram size (2 for bigram, 3 for trigram)
        
    Returns:
        Overlap score (number of common n-grams / total unique n-grams)
    """
    def get_ngrams(text: str, n: int) -> set:
        """Extract n-grams from text."""
        text = text.lower().replace(" ", "")
        if len(text) < n:
            return set()
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    
    return len(intersection) / len(union) if union else 0.0


def extract_repos_from_papers(papers: List[Source]) -> List[Source]:
    """
    Extract additional GitHub repositories from papers by reading their PDF pages.
    For each paper, selects the most relevant GitHub repo based on n-gram overlap.
    
    Args:
        papers: List of paper sources to extract repos from
        
    Returns:
        List of GitHub repositories found in papers (one per paper)
    """
    github_repos = []
    
    for paper in papers:
        try:
            url = paper.pdf_url or paper.url
            if not url:
                continue
                
            result = read_page(url)
            if not isinstance(result, dict):
                continue
                
            md = result.get("md", {})
            if not isinstance(md, dict):
                continue
                
            links = md.get("links", [])
            if not links:
                continue
            
            # Collect and normalize all GitHub links
            normalized_repos = []
            for link in links:
                if "github.com" not in link:
                    continue
                
                normalized_url = normalize_github_url(link)
                if normalized_url:
                    # Extract repo name (last field)
                    repo_name = normalized_url.split("/")[-1]
                    normalized_repos.append((normalized_url, repo_name))
            
            if not normalized_repos:
                continue
            
            # If only one repo, use it directly
            if len(normalized_repos) == 1:
                normalized_url, _ = normalized_repos[0]
                github_repo = Source(
                    title=f"Paper Repository: {paper.title}",
                    url=normalized_url,
                    source_type=SourceType.CODE,
                    platform=Platform.GITHUB,
                    description=f"Repository mentioned in paper: {paper.title}",
                    metadata={"paper_title": paper.title, "paper_url": url},
                )
                github_repos.append(github_repo)
                continue
            
            # Multiple repos: select the one with highest n-gram overlap
            paper_title = paper.title.lower()
            best_repo = None
            best_score = -1.0
            
            for normalized_url, repo_name in normalized_repos:
                # Compute 2-gram and 3-gram overlap
                score_2gram = compute_ngram_overlap(repo_name, paper_title, 2)
                score_3gram = compute_ngram_overlap(repo_name, paper_title, 3)
                # Use average of both scores
                combined_score = (score_2gram + score_3gram) / 2.0
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_repo = normalized_url
            
            if best_repo:
                github_repo = Source(
                    title=f"Paper Repository: {paper.title}",
                    url=best_repo,
                    source_type=SourceType.CODE,
                    platform=Platform.GITHUB,
                    description=f"Repository mentioned in paper: {paper.title}",
                    metadata={"paper_title": paper.title, "paper_url": url},
                )
                github_repos.append(github_repo)
                
        except Exception as e:
            logger.warning(f"Failed to read page for paper {paper.url}: {e}")
            continue
    
    return github_repos


