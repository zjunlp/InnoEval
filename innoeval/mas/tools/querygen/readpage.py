import os
import re
import json
import time
import logging
from typing import Dict, List, Optional

import requests
import tiktoken
import dspy

logger = logging.getLogger(__name__)


def read_with_jina(url: str, timeout: int = 50, retries: int = 3) -> str:
    api_key = os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if resp.status_code == 200:
                text = resp.text or ""
                if text.strip():
                    return text
                return "[readpage] Empty content."
            last_err = f"status={resp.status_code} body={resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5)
    return f"[readpage] Failed to read page: {last_err}"


def parse_markdown_metadata(text: str) -> Dict[str, Optional[str]]:
    title = None
    headings: List[str] = []
    links: List[str] = []

    for line in text.splitlines():
        if not title:
            m = re.match(r"^#\s+(.*)$", line.strip())
            if m:
                title = m.group(1).strip()
        hm = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if hm:
            headings.append(hm.group(2).strip())
        for url in re.findall(r"https?://[^\s)]+", line):
            links.append(url)
        for mlink in re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", line):
            links.append(mlink)

    if not title:
        mt = re.search(r"^\s*title\s*:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
        if mt:
            title = mt.group(1).strip()

    return {
        "title": title,
        "headings": headings,
        "links": links,
    }


def read_page(url: str) -> Dict[str, object]:
    logger.info(f"read_page: Starting to read page - URL: {url}")
    
    raw = read_with_jina(url)
    if raw.startswith("[readpage] Failed"):
        logger.error(f"read_page: Failed to read page from Jina - URL: {url}, error: {raw}")
        return {"raw": raw, "md": {}}
    
    logger.info(f"read_page: Successfully retrieved raw content, length: {len(raw)} characters for URL: {url}")

    md = parse_markdown_metadata(raw)
    logger.info(f"read_page: Parsed metadata - title: '{md.get('title')}', headings: {len(md.get('headings', []))}, links: {len(md.get('links', []))} for URL: {url}")

    logger.info(f"read_page: Successfully completed reading page for URL: {url}")
    return {"raw": raw, "md": md}

