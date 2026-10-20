"""Document and URL ingestion, re-exported for backwards-compatible imports."""

from app.ingestion.chunking import chunk_text
from app.ingestion.extractors import clean_html_to_text, extract_text_from_file
from app.ingestion.fetcher import fetch_url_safely
from app.ingestion.security import SSRFBlockedError, validate_url_against_ssrf
from app.ingestion.service import ingest_file_stream, ingest_url_stream
from app.ingestion.sse import sse_event

__all__ = [
    "SSRFBlockedError",
    "chunk_text",
    "clean_html_to_text",
    "extract_text_from_file",
    "fetch_url_safely",
    "ingest_file_stream",
    "ingest_url_stream",
    "sse_event",
    "validate_url_against_ssrf",
]
