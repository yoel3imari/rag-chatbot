"""Bounded HTTP fetching with per-redirect SSRF validation."""

import asyncio
import urllib.parse
from typing import Tuple

import httpx

from app.config import MAX_UPLOAD_BYTES
from app.ingestion.security import validate_url_against_ssrf


async def fetch_url_safely(url: str, max_redirects: int = 5) -> Tuple[str, int]:
    """
    Fetches a URL with SSRF validation on the initial target and every redirect hop.
    Streams the response body incrementally and enforces MAX_UPLOAD_BYTES during the
    download, so a hostile server cannot exhaust memory before the size check applies.
    """
    current_url = await asyncio.to_thread(validate_url_against_ssrf, url)

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=False,
        headers={"User-Agent": "CompanyRAGBot/1.0 (+https://localhost)"},
    ) as client:
        for _ in range(max_redirects + 1):
            async with client.stream("GET", current_url) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError(
                            f"Redirect ({response.status_code}) missing Location header."
                        )
                    next_url = urllib.parse.urljoin(current_url, location)
                    current_url = await asyncio.to_thread(
                        validate_url_against_ssrf, next_url
                    )
                    continue

                response.raise_for_status()
                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > MAX_UPLOAD_BYTES:
                        raise ValueError(
                            f"URL response size exceeds max allowed size ({MAX_UPLOAD_BYTES} bytes)."
                        )
                charset = response.charset_encoding or "utf-8"
                return buffer.decode(charset, errors="replace"), len(buffer)

        raise ValueError(f"Exceeded maximum redirects ({max_redirects}).")
