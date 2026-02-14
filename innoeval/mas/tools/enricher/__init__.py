"""
General enrichment tools: paper extraction, web/code report generation.
Keep decoupled from agents; only expose async enrichment functions in use.
"""

from .enricher import (
    enrich_papers_with_extraction,
    enrich_web_with_reports,
    enrich_code_with_rawtext,
    enrich_code_with_repo,
)

__all__ = [
    "enrich_papers_with_extraction",
    "enrich_web_with_reports",
    "enrich_code_with_rawtext",
    "enrich_code_with_repo",
]


