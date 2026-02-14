"""
XML Parser for GROBID TEI XML output

This module provides a parser to extract structured information from
GROBID's TEI XML format.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional


class GrobidXMLParser:
    """
    Parser for GROBID-generated TEI XML documents

    Capabilities:
    - Extract bibliography (reference list)
    - Extract citations from text
    - Extract sections and content
    """

    # TEI namespace
    TEI_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def __init__(self, xml_content: str):
        """
        Initialize parser with XML content

        Args:
            xml_content: GROBID TEI XML content as string
        """
        self.xml_content = xml_content
        self.root = None
        self._parse_xml()

    def _parse_xml(self):
        """Parse XML content and set root element"""
        try:
            self.root = ET.fromstring(self.xml_content)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            # Try wrapping as fragment
            try:
                wrapped = f'<root xmlns="http://www.tei-c.org/ns/1.0">{self.xml_content}</root>'
                self.root = ET.fromstring(wrapped)
            except ET.ParseError:
                raise ValueError(f"Unable to parse XML content: {e}")

    def extract_bibliography(self) -> List[Dict[str, Any]]:
        """
        Extract complete bibliography (reference list) from XML

        GROBID stores references in <back><listBibl> section.
        Each reference is a <biblStruct> element.

        Returns:
            List of reference dictionaries with fields:
            - id: Reference ID (e.g., 'b5')
            - title: Paper title
            - authors: List of author names
            - year: Publication year
            - venue: Journal/conference name
            - pages: Page range
            - doi: DOI identifier
            - url: URL if available
        """
        if self.root is None:
            return []

        bibliography = []

        # Find all biblStruct elements in listBibl
        bibl_entries = self.root.findall(
            './/tei:listBibl/tei:biblStruct',
            self.TEI_NS
        )

        print(f"Found {len(bibl_entries)} bibliography entries")

        for bibl in bibl_entries:
            entry = self._parse_bibl_struct(bibl)
            if entry:
                bibliography.append(entry)

        return bibliography

    def _parse_bibl_struct(self, bibl: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Parse a single biblStruct element

        Args:
            bibl: biblStruct XML element

        Returns:
            Dictionary with reference information
        """
        entry = {
            'id': bibl.get('{http://www.w3.org/XML/1998/namespace}id', ''),
            'title': '',
            'authors': [],
            'year': '',
            'venue': '',
            'pages': '',
            'doi': '',
            'url': ''
        }

        # Extract title (level="a" means article title)
        title_elem = bibl.find('.//tei:title[@level="a"]', self.TEI_NS)
        if title_elem is not None and title_elem.text:
            entry['title'] = title_elem.text.strip()

        # Extract authors
        authors = bibl.findall('.//tei:author/tei:persName', self.TEI_NS)
        for author in authors:
            author_name = self._parse_person_name(author)
            if author_name:
                entry['authors'].append(author_name)

        # Extract publication year
        date_elem = bibl.find('.//tei:date', self.TEI_NS)
        if date_elem is not None:
            # Prefer 'when' attribute (standard format)
            entry['year'] = date_elem.get('when', '') or date_elem.text or ""
            # Extract just the year if full date provided
            if entry['year'] and len(entry['year']) > 4:
                entry['year'] = entry['year'][:4]

        # Extract venue (journal or conference)
        venue_elem = (
            bibl.find('.//tei:title[@level="j"]', self.TEI_NS) or  # journal
            bibl.find('.//tei:title[@level="m"]', self.TEI_NS)     # monograph/conference
        )
        if venue_elem is not None and venue_elem.text:
            entry['venue'] = venue_elem.text.strip()

        # Extract page range
        pages_elem = bibl.find('.//tei:biblScope[@unit="page"]', self.TEI_NS)
        if pages_elem is not None:
            from_page = pages_elem.get('from', '')
            to_page = pages_elem.get('to', '')
            if from_page and to_page:
                entry['pages'] = f"{from_page}-{to_page}"
            elif pages_elem.text:
                entry['pages'] = pages_elem.text.strip()

        # Extract DOI
        doi_elem = bibl.find('.//tei:idno[@type="DOI"]', self.TEI_NS)
        if doi_elem is not None and doi_elem.text:
            entry['doi'] = doi_elem.text.strip()

        # Extract URL/arXiv
        url_elem = bibl.find('.//tei:idno[@type="arXiv"]', self.TEI_NS)
        if url_elem is not None and url_elem.text:
            arxiv_id = url_elem.text.strip()
            entry['url'] = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            # Try generic URL
            url_elem = bibl.find('.//tei:ptr[@type="web"]', self.TEI_NS)
            if url_elem is not None:
                entry['url'] = url_elem.get('target', '')

        return entry if entry['title'] else None

    def _parse_person_name(self, persName: ET.Element) -> str:
        """
        Parse a persName element to extract full name

        Args:
            persName: persName XML element

        Returns:
            Full name as string
        """
        forename = persName.find('tei:forename', self.TEI_NS)
        surname = persName.find('tei:surname', self.TEI_NS)

        name_parts = []
        if forename is not None and forename.text:
            name_parts.append(forename.text.strip())
        if surname is not None and surname.text:
            name_parts.append(surname.text.strip())

        return " ".join(name_parts)
