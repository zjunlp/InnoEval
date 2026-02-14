"""
GROBID References Extraction Tool

This package provides tools to extract references from PDF files using GROBID.
"""

from .reference_extractor import extract_references_from_pdf, GrobidReferenceExtractor

__all__ = [
    'extract_references_from_pdf',
    'GrobidReferenceExtractor'
]
