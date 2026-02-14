"""
PDF Processor using GROBID

This module handles PDF processing workflow including caching and error handling.
"""

import os
import shutil
from typing import Dict, Any, Optional

try:
    from .grobid_client import GrobidClient
except ImportError:
    from grobid_client import GrobidClient


class PDFProcessor:
    """
    Process PDF files with GROBID and manage caching
    """

    def __init__(
        self,
        grobid_server: str = "http://127.0.0.1:8070",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize PDF processor

        Args:
            grobid_server: GROBID server URL
            cache_dir: Directory to cache XML outputs (optional)
        """
        self.client = GrobidClient(grobid_server=grobid_server)
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def process_pdf_with_grobid(
        self,
        pdf_path: str,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process a PDF file and generate TEI XML output

        Workflow:
        1. Check if cached XML exists
        2. If not, process PDF with GROBID
        3. Save XML to cache
        4. Return result with XML content

        Args:
            pdf_path: Path to the PDF file
            force_reprocess: Force reprocessing even if cache exists

        Returns:
            Dictionary with:
            - success: bool
            - xml_content: str (TEI XML content)
            - xml_path: str (path to cached XML, if cache_dir is set)
            - pdf_file: str (PDF filename)
            - status: str (cached/processed/error)
            - error: str (error message if failed)
        """
        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}",
                "status": "error"
            }

        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]

        result = {
            "pdf_file": pdf_filename,
            "success": False,
            "xml_content": None,
            "xml_path": None,
            "status": "pending"
        }

        # Check cache
        xml_path = None
        if self.cache_dir:
            xml_path = os.path.join(
                self.cache_dir,
                f"{pdf_name}.grobid.tei.xml"
            )
            result["xml_path"] = xml_path

            if os.path.exists(xml_path) and not force_reprocess:
                # Load from cache
                try:
                    with open(xml_path, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                    result["xml_content"] = xml_content
                    result["success"] = True
                    result["status"] = "cached"
                    print(f"✓ Loaded cached XML for {pdf_filename}")
                    return result
                except Exception as e:
                    print(f"Failed to load cached XML: {e}")
                    # Continue to reprocess

        # Process with GROBID
        print(f"Processing {pdf_filename} with GROBID...")
        xml_content = self.client.process_pdf(pdf_path)

        if xml_content:
            result["xml_content"] = xml_content
            result["success"] = True
            result["status"] = "processed"

            # Save to cache
            if xml_path:
                try:
                    with open(xml_path, 'w', encoding='utf-8') as f:
                        f.write(xml_content)
                    print(f"✓ Saved XML to cache: {xml_path}")
                except Exception as e:
                    print(f"Warning: Failed to save XML to cache: {e}")

            return result
        else:
            result["error"] = "GROBID processing failed"
            result["status"] = "error"
            return result
