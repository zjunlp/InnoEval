"""
GROBID Client for processing PDF files

This module provides a client to interact with GROBID service.
"""

import os
import requests
import time
from typing import Optional, Dict, Any


class GrobidClient:
    """
    Client for interacting with GROBID service

    GROBID is a machine learning library for extracting, parsing and
    restructuring raw documents such as PDF into structured TEI-encoded documents.
    """

    def __init__(
        self,
        grobid_server: str = "http://127.0.0.1:8070",
        timeout: int = 60,
        sleep_time: int = 5,
        check_server: bool = True
    ):
        """
        Initialize GROBID client

        Args:
            grobid_server: GROBID server URL
            timeout: Request timeout in seconds
            sleep_time: Sleep time between requests
            check_server: Whether to check server availability on init
        """
        self.grobid_server = grobid_server.rstrip('/')
        self.timeout = timeout
        self.sleep_time = sleep_time

        if check_server:
            self._check_server()

    def _check_server(self) -> bool:
        """
        Check if GROBID server is available

        Returns:
            True if server is available, False otherwise
        """
        try:
            response = requests.get(
                f"{self.grobid_server}/api/isalive",
                timeout=5
            )
            if response.status_code == 200:
                print(f"✓ GROBID server is available at {self.grobid_server}")
                return True
            else:
                print(f"✗ GROBID server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to GROBID server: {e}")
            return False

    def process_pdf(
        self,
        pdf_path: str,
        service: str = "processFulltextDocument"
    ) -> Optional[str]:
        """
        Process a PDF file with GROBID

        Args:
            pdf_path: Path to the PDF file
            service: GROBID service to use (default: processFulltextDocument)
                    Options: processFulltextDocument, processReferences, etc.

        Returns:
            TEI XML content as string, or None if processing failed
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        url = f"{self.grobid_server}/api/{service}"

        try:
            with open(pdf_path, 'rb') as pdf_file:
                files = {
                    'input': (
                        os.path.basename(pdf_path),
                        pdf_file,
                        'application/pdf'
                    )
                }

                response = requests.post(
                    url,
                    files=files,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 503:
                    print(f"GROBID service unavailable (503), retrying...")
                    time.sleep(self.sleep_time)
                    return self.process_pdf(pdf_path, service)
                else:
                    print(f"GROBID processing failed with status {response.status_code}")
                    print(f"Response: {response.text[:200]}")
                    return None

        except requests.exceptions.Timeout:
            print(f"Request timeout after {self.timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def process_references(self, pdf_path: str) -> Optional[str]:
        """
        Process only the references section of a PDF

        Args:
            pdf_path: Path to the PDF file

        Returns:
            TEI XML content with references, or None if processing failed
        """
        return self.process_pdf(pdf_path, service="processReferences")
