"""
Reference Extractor - Main tool for extracting references from PDFs

This module provides the main interface for extracting references from PDF files
using GROBID and returning them as (title, url) tuples.
"""

import os
from typing import List, Tuple, Dict, Any, Optional

try:
    from .pdf_processor import PDFProcessor
    from .xml_parser import GrobidXMLParser
except ImportError:
    from pdf_processor import PDFProcessor
    from xml_parser import GrobidXMLParser


class GrobidReferenceExtractor:
    """
    Extract references from PDF files using GROBID

    This class provides a high-level interface to:
    1. Process PDF files with GROBID
    2. Parse the resulting TEI XML
    3. Extract references as (title, url) tuples
    """

    def __init__(
        self,
        grobid_server: str = "http://127.0.0.1:8070",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize reference extractor

        Args:
            grobid_server: GROBID server URL (default: http://127.0.0.1:8070)
            cache_dir: Directory to cache XML outputs (optional)
        """
        self.processor = PDFProcessor(
            grobid_server=grobid_server,
            cache_dir=cache_dir
        )

    def extract_references(
        self,
        pdf_path: str,
        force_reprocess: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Extract references from a PDF file

        Args:
            pdf_path: Path to the PDF file
            force_reprocess: Force reprocessing even if cache exists

        Returns:
            List of (title, url) tuples for each reference
            URL may be empty string if not available
        """
        # Step 1: Process PDF with GROBID
        result = self.processor.process_pdf_with_grobid(
            pdf_path,
            force_reprocess=force_reprocess
        )

        if not result["success"]:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return []

        xml_content = result["xml_content"]
        if not xml_content:
            print("Error: No XML content returned")
            return []

        # Step 2: Parse XML and extract bibliography
        try:
            parser = GrobidXMLParser(xml_content)
            bibliography = parser.extract_bibliography()
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return []

        # Step 3: Convert to (title, url) tuples
        references = []
        for entry in bibliography:
            title = entry.get('title', '').strip()
            if not title:
                continue

            # Try to construct URL from available information
            url = self._construct_url(entry)

            references.append((title, url))

        return references

    def extract_references_detailed(
        self,
        pdf_path: str,
        force_reprocess: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract detailed reference information from a PDF file

        Args:
            pdf_path: Path to the PDF file
            force_reprocess: Force reprocessing even if cache exists

        Returns:
            List of dictionaries with detailed reference information:
            - title: Paper title
            - url: URL (if available)
            - authors: List of author names
            - year: Publication year
            - venue: Journal/conference name
            - doi: DOI identifier
        """
        # Step 1: Process PDF with GROBID
        result = self.processor.process_pdf_with_grobid(
            pdf_path,
            force_reprocess=force_reprocess
        )

        if not result["success"]:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return []

        xml_content = result["xml_content"]
        if not xml_content:
            print("Error: No XML content returned")
            return []

        # Step 2: Parse XML and extract bibliography
        try:
            parser = GrobidXMLParser(xml_content)
            bibliography = parser.extract_bibliography()
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return []

        # Step 3: Enhance with URLs
        for entry in bibliography:
            if not entry.get('url'):
                entry['url'] = self._construct_url(entry)

        return bibliography

    def _construct_url(self, entry: Dict[str, Any]) -> str:
        """
        Construct URL from reference entry

        Priority:
        1. Existing URL field (arXiv, etc.)
        2. DOI -> https://doi.org/{doi}
        3. Empty string if no URL available

        Args:
            entry: Reference entry dictionary

        Returns:
            URL string (may be empty)
        """
        # Check if URL already exists
        if entry.get('url'):
            return entry['url']

        # Try to construct from DOI
        doi = entry.get('doi', '').strip()
        if doi:
            # Remove any existing URL prefix from DOI
            doi = doi.replace('https://doi.org/', '')
            doi = doi.replace('http://dx.doi.org/', '')
            return f"https://doi.org/{doi}"

        # No URL available
        return ""


def extract_references_from_pdf(
    pdf_path: str,
    grobid_server: str = "http://127.0.0.1:8070",
    cache_dir: Optional[str] = None,
    force_reprocess: bool = False
) -> List[Tuple[str, str]]:
    """
    Convenience function to extract references from a PDF file

    Args:
        pdf_path: Path to the PDF file
        grobid_server: GROBID server URL (default: http://127.0.0.1:8070)
        cache_dir: Directory to cache XML outputs (optional)
        force_reprocess: Force reprocessing even if cache exists

    Returns:
        List of (title, url) tuples for each reference

    Example:
        >>> references = extract_references_from_pdf(
        ...     "/path/to/paper.pdf",
        ...     grobid_server="http://127.0.0.1:2344"
        ... )
        >>> for title, url in references:
        ...     print(f"{title}: {url}")
    """
    extractor = GrobidReferenceExtractor(
        grobid_server=grobid_server,
        cache_dir=cache_dir
    )
    return extractor.extract_references(pdf_path, force_reprocess)
