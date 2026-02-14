"""
Timestamp utilities for GitHub repositories and web pages.

Provides functions to extract repository information, fetch timestamps from GitHub API,
and extract timestamps from web page raw text.
"""

import logging
import os
import re
import requests
from datetime import datetime
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def extract_repo_info(url: str) -> Optional[Tuple[str, str]]:
    """
    Extract owner and repo name from GitHub URL.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Tuple of (owner, repo) or None if invalid
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
    
    owner = parts[0]
    repo = parts[1]
    return (owner, repo)


def get_repo_timestamps(url: str) -> Optional[Dict[str, str]]:
    """
    Get repository creation and update timestamps from GitHub API.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Dictionary with 'created_at', 'updated_at', and 'pushed_at' keys, or None if failed
    """
    repo_info = extract_repo_info(url)
    if not repo_info:
        logger.warning(f"Could not extract repo info from URL: {url}")
        return None
    
    owner, repo = repo_info
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    
    # Create session with GitHub token if available
    session = requests.Session()
    github_token = os.getenv('GITHUB_AI_TOKEN')
    if github_token:
        session.headers.update({
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        })
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        response = session.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'pushed_at': data.get('pushed_at'),
            }
        elif response.status_code == 404:
            logger.warning(f"Repository not found: {owner}/{repo}")
        elif response.status_code == 403:
            logger.warning(f"Rate limit or access denied for: {owner}/{repo}")
        else:
            logger.warning(f"GitHub API returned status {response.status_code} for {owner}/{repo}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching repo timestamps for {url}: {e}")
    
    return None


def extract_date_from_raw_text(raw_text: str) -> Optional[str]:
    """
    Extract date in YYYY-MM-DD format from raw text.
    
    Looks for patterns like:
    - Published Time: 2025-11-29T15:16:19Z
    - Published Time: Wed, 03 Dec 2025 13:08:16 GMT
    - Published Time: 2023-10-31T10:26:10+00:00
    - Published Time: 2024-06-02T11:03:45+00:00
    
    Args:
        raw_text: Raw text content from web page
        
    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    if not raw_text:
        return None
    
    # Pattern 1: ISO format with T separator: 2025-11-29T15:16:19Z or 2023-10-31T10:26:10+00:00
    iso_pattern = r'Published Time:\s*(\d{4}-\d{2}-\d{2})T'
    match = re.search(iso_pattern, raw_text)
    if match:
        return match.group(1)
    
    # Pattern 2: RFC 2822 format: Wed, 03 Dec 2025 13:08:16 GMT
    rfc_pattern = r'Published Time:\s*\w+,\s*(\d{1,2})\s+(\w{3})\s+(\d{4})'
    match = re.search(rfc_pattern, raw_text)
    if match:
        day = match.group(1).zfill(2)
        month_str = match.group(2)
        year = match.group(3)
        
        # Convert month abbreviation to number
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        month = month_map.get(month_str)
        if month:
            return f"{year}-{month}-{day}"
    
    # Pattern 3: Direct YYYY-MM-DD format in the text (look for "Published Time:" context)
    direct_pattern = r'Published Time:\s*(\d{4}-\d{2}-\d{2})'
    match = re.search(direct_pattern, raw_text)
    if match:
        return match.group(1)
    
    # Pattern 4: Look for any YYYY-MM-DD pattern near "Published" keyword
    published_context_pattern = r'Published[^:]*:\s*.*?(\d{4}-\d{2}-\d{2})'
    match = re.search(published_context_pattern, raw_text, re.IGNORECASE)
    if match:
        date_str = match.group(1)
        # Validate it's a reasonable date
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            pass
    
    return None


def is_date_in_range(date_str: Optional[str], before: Optional[str] = None, after: Optional[str] = None) -> bool:
    """
    Check if a date string (YYYY-MM-DD) is within the specified range.
    
    Args:
        date_str: Date string in YYYY-MM-DD format (can be None)
        before: Optional upper bound date (YYYY-MM-DD format)
        after: Optional lower bound date (YYYY-MM-DD format)
        
    Returns:
        True if date is within range (or no filters specified), False otherwise
    """
    if date_str is None:
        # If no date available, include it (don't filter out)
        return True
    
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        # Invalid date format, include it
        return True
    
    if after:
        try:
            after_date = datetime.strptime(after, '%Y-%m-%d')
            if date < after_date:
                return False
        except ValueError:
            pass
    
    if before:
        try:
            before_date = datetime.strptime(before, '%Y-%m-%d')
            if date > before_date:
                return False
        except ValueError:
            pass
    
    return True

