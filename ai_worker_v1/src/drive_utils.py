"""
Utilities to normalize common Google Drive sharing URLs into direct downloads.
"""

import logging
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse


logger = logging.getLogger("OneFrame.Drive")

_DRIVE_HOST_TOKENS = ("drive.google.com", "docs.google.com")
_DRIVE_PATH_PATTERNS = (
    re.compile(r"/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"/d/([a-zA-Z0-9_-]+)"),
)


def is_google_drive_url(url: str) -> bool:
    if not url:
        return False

    try:
        hostname = (urlparse(url.strip()).hostname or "").lower()
    except ValueError:
        return False

    return any(token in hostname for token in _DRIVE_HOST_TOKENS)


def extract_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None

    trimmed = url.strip()
    if not trimmed:
        return None

    for pattern in _DRIVE_PATH_PATTERNS:
        match = pattern.search(trimmed)
        if match:
            return match.group(1)

    parsed = urlparse(trimmed)
    query_params = parse_qs(parsed.query)
    if query_params.get("id"):
        return query_params["id"][0]

    return None


def normalize_drive_url(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("Empty URL")

    trimmed = url.strip()
    logger.info("Original URL received: %s", trimmed)

    if not is_google_drive_url(trimmed):
        logger.info("Non-Google-Drive URL detected; using original URL without changes.")
        return trimmed

    file_id = extract_drive_file_id(trimmed)
    if not file_id:
        message = f"Could not extract Google Drive file id from URL: {trimmed}"
        logger.error(message)
        raise ValueError(message)

    normalized_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    logger.info("Detected Google Drive file_id: %s", file_id)
    logger.info("Normalized Google Drive URL: %s", normalized_url)
    logger.info("Google Drive URL normalized successfully")
    return normalized_url
