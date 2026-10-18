import asyncio
import io
import ipaddress
import json
import re
import socket
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import httpx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import MAX_UPLOAD_BYTES, UPLOADS_DIR
from app.embeddings import embedding_engine
from app.store import store
from app.vector_store import vector_store


class SSRFBlockedError(ValueError):
    """Raised when a URL resolves to a private, loopback, link-local, or reserved address."""


def validate_url_against_ssrf(url: str) -> str:
    """
    Validates that `url` uses http/https and does not target localhost, private LAN ranges,
    loopback (127.0.0.0/8, ::1), or cloud metadata link-local addresses (169.254.169.254).
    """
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise SSRFBlockedError(f"Blocked URL scheme '{parsed.scheme}'. Only http and https are allowed.")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise SSRFBlockedError("URL is missing a valid hostname.")

    if hostname in ("localhost", "localhost.localdomain") or hostname.endswith((".local", ".internal", ".localhost")):
        raise SSRFBlockedError(f"Blocked local/internal hostname: {hostname}")

    # Check if literal IP address first
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            raise SSRFBlockedError(f"Blocked restricted IP address: {ip_obj}")
        return url.strip()
    except ValueError as e:
        if isinstance(e, SSRFBlockedError):
            raise

    # Resolve DNS records and inspect every resolved IP
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname '{hostname}': {exc}") from exc

    if not addr_info:
        raise ValueError(f"No DNS records found for hostname '{hostname}'.")

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        # Strip IPv6 scope ID if present
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        ip_obj = ipaddress.ip_address(ip_str)
        # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
            ip_obj = ip_obj.ipv4_mapped
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            raise SSRFBlockedError(
                f"Blocked SSRF attempt: '{hostname}' resolved to restricted IP {ip_obj}."
            )

    return url.strip()


async def fetch_url_safely(url: str, max_redirects: int = 5) -> Tuple[str, int]:
    """
    Fetches a URL with SSRF validation on the initial target and every redirect hop.
    Streams the response body incrementally and enforces MAX_UPLOAD_BYTES *during* the
    download, so a hostile server cannot exhaust memory before the size check applies.
    """
    current_url = await asyncio.to_thread(validate_url_against_ssrf, url)

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=False,
        headers={"User-Agent": "CompanyRAGBot/1.0 (+https://localhost)"},
    ) as client:
        for _ in range(max_redirects + 1):
            async with client.stream("GET", current_url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise ValueError(f"Redirect ({resp.status_code}) missing Location header.")
                    next_url = urllib.parse.urljoin(current_url, location)
                    current_url = await asyncio.to_thread(validate_url_against_ssrf, next_url)
                    continue

                resp.raise_for_status()
                buffer = bytearray()
                async for chunk in resp.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > MAX_UPLOAD_BYTES:
                        raise ValueError(
                            f"URL response size exceeds max allowed size ({MAX_UPLOAD_BYTES} bytes)."
                        )
                charset = resp.charset_encoding or "utf-8"
                return buffer.decode(charset, errors="replace"), len(buffer)

        raise ValueError(f"Exceeded maximum redirects ({max_redirects}).")


def clean_html_to_text(html_content: str, url: str = "") -> str:
    """Strips boilerplate HTML, navigation trees, scripts, and redundant whitespace to maximize token density."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "noscript",
            "svg",
            "form",
            "iframe",
            "button",
        ]
    ):
        tag.decompose()

    for el in soup.find_all(
        attrs={
            "role": re.compile(r"^(navigation|banner|contentinfo|complementary|search)$", re.I)
        }
    ):
        el.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    main_node = soup.find("main") or soup.find("article") or soup.body or soup

    lines: List[str] = []
    if title:
        lines.append(f"# {title}\n")

    for elem in main_node.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code", "blockquote", "td", "th"]):
        txt = elem.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt or len(txt) < 3:
            continue
        if elem.name in ("h1", "h2", "h3", "h4"):
            prefix = "#" * int(elem.name[1])
            lines.append(f"\n{prefix} {txt}\n")
        elif elem.name == "li":
            lines.append(f"- {txt}")
        else:
            lines.append(txt)

    if len(lines) <= 2:
        raw_text = main_node.get_text("\n", strip=True)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        raw_text = re.sub(r"[ \t]+", " ", raw_text)
        return raw_text.strip()

    dedup_lines: List[str] = []
    for line in lines:
        if not dedup_lines or dedup_lines[-1] != line:
            dedup_lines.append(line)

    joined = "\n".join(dedup_lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extracts clean text from .pdf, .txt, or .md files."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text: List[str] = []
        for idx, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            t = re.sub(r"[ \t]+", " ", t).strip()
            if t:
                pages_text.append(f"[Page {idx + 1}]\n{t}")
        return "\n\n".join(pages_text)
    else:
        text = file_bytes.decode("utf-8", errors="replace")
        return re.sub(r"\r\n", "\n", text).strip()


def chunk_text(text: str, chunk_size: int = 750, chunk_overlap: int = 120) -> List[str]:
    """Splits text into overlapping semantic chunks using LangChain's RecursiveCharacterTextSplitter."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def _sse_event(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def ingest_file_stream(
    filename: str,
    file_bytes: bytes,
    scope: str = "global",
    session_id: str = "",
    user_id: str = "default",
    idempotency_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streams real-time SSE ingestion events while parsing, embedding with CrispEmbed, and indexing into LanceDB."""
    safe_name = Path(filename).name
    file_size = len(file_bytes)
    save_path = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    save_path.write_bytes(file_bytes)

    doc = store.create_document_record(
        name=safe_name,
        source_type="file",
        scope=scope,
        session_id=session_id,
        file_size=file_size,
        raw_path_or_url=str(save_path),
        user_id=user_id,
    )
    doc_id = doc["id"]

    yield _sse_event(
        {
            "doc_id": doc_id,
            "name": safe_name,
            "source_type": "file",
            "scope": scope,
            "status": "QUEUED",
            "progress": 10,
            "message": f"Queued {safe_name} ({file_size // 1024 or 1} KB)",
        }
    )
    await asyncio.sleep(0.02)

    try:
        store.update_document_status(doc_id, "PARSING")
        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": safe_name,
                "source_type": "file",
                "scope": scope,
                "status": "PARSING",
                "progress": 30,
                "message": f"Parsing text from {safe_name}...",
            }
        )
        text = await asyncio.to_thread(extract_text_from_file, file_bytes, safe_name)
        if not text.strip():
            raise ValueError("No extractable text found in document.")

        chunks = await asyncio.to_thread(chunk_text, text)
        if not chunks:
            raise ValueError("Document produced 0 text chunks.")

        settings = store.get_settings(user_id=user_id)
        emb_model = settings.get("embedding_model", "all-MiniLM-L6-v2")
        store.update_document_status(doc_id, "PROCESSING", chunk_count=len(chunks))
        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": safe_name,
                "source_type": "file",
                "scope": scope,
                "status": "EMBEDDING",
                "progress": 65,
                "message": f"Embedding {len(chunks)} chunks via CrispEmbed ({emb_model})...",
            }
        )
        vectors = await asyncio.to_thread(embedding_engine.embed_batch, chunks, emb_model)
        resolved_model = embedding_engine.model_name

        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": safe_name,
                "source_type": "file",
                "scope": scope,
                "status": "INDEXING",
                "progress": 85,
                "message": f"Indexing {len(chunks)} vectors + FTS into LanceDB...",
            }
        )
        records = [
            {
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source_name": safe_name,
                "source_type": "file",
                "scope": scope,
                "session_id": session_id or (scope if scope != "global" else ""),
                "user_id": user_id,
                "embedding_model": resolved_model,
                "chunk_index": i,
                "text": chunk_str,
                "vector": vec,
            }
            for i, (chunk_str, vec) in enumerate(zip(chunks, vectors))
        ]
        await asyncio.to_thread(vector_store.add_chunks, records)

        updated_doc = store.update_document_status(doc_id, "INDEXED", chunk_count=len(chunks))
        final_sse = _sse_event(
            {
                "doc_id": doc_id,
                "name": safe_name,
                "source_type": "file",
                "scope": scope,
                "status": "INDEXED",
                "progress": 100,
                "chunk_count": len(chunks),
                "document": updated_doc,
                "message": f"Indexed {safe_name} ({len(chunks)} chunks)",
            }
        )
        if idempotency_key:
            store.save_idempotency(idempotency_key, final_sse, ttl_seconds=300.0)
        yield final_sse
    except Exception as exc:
        err_msg = str(exc)
        store.update_document_status(doc_id, "FAILED", error_message=err_msg)
        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": safe_name,
                "source_type": "file",
                "scope": scope,
                "status": "FAILED",
                "progress": 100,
                "error": err_msg,
                "message": f"Failed to ingest {safe_name}: {err_msg}",
            }
        )


async def ingest_url_stream(
    url: str,
    scope: str = "global",
    session_id: str = "",
    user_id: str = "default",
    existing_doc_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streams real-time SSE events while scraping a URL (with SSRF protection), stripping HTML boilerplate, embedding with CrispEmbed, and indexing into LanceDB."""
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    if existing_doc_id:
        doc = store.get_document(existing_doc_id)
        if not doc:
            doc = store.create_document_record(
                name=clean_url,
                source_type="url",
                scope=scope,
                session_id=session_id,
                raw_path_or_url=clean_url,
                user_id=user_id,
            )
        else:
            await asyncio.to_thread(vector_store.delete_by_doc_ids, [existing_doc_id])
            store.update_document_status(existing_doc_id, "QUEUED")
    else:
        doc = store.create_document_record(
            name=clean_url,
            source_type="url",
            scope=scope,
            session_id=session_id,
            raw_path_or_url=clean_url,
            user_id=user_id,
        )

    doc_id = doc["id"]

    yield _sse_event(
        {
            "doc_id": doc_id,
            "name": clean_url,
            "source_type": "url",
            "scope": scope,
            "status": "QUEUED",
            "progress": 10,
            "message": f"Validating & fetching {clean_url}...",
        }
    )

    try:
        store.update_document_status(doc_id, "PARSING")
        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": clean_url,
                "source_type": "url",
                "scope": scope,
                "status": "PARSING",
                "progress": 30,
                "message": f"Scraping & stripping HTML boilerplate from {clean_url}...",
            }
        )

        raw_html, byte_len = await fetch_url_safely(clean_url)

        text = await asyncio.to_thread(clean_html_to_text, raw_html, clean_url)
        if not text.strip():
            raise ValueError("No readable text content found at URL.")

        chunks = await asyncio.to_thread(chunk_text, text)
        if not chunks:
            raise ValueError("URL content produced 0 text chunks.")

        settings = store.get_settings(user_id=user_id)
        emb_model = settings.get("embedding_model", "all-MiniLM-L6-v2")
        store.update_document_status(doc_id, "PROCESSING", chunk_count=len(chunks), file_size=byte_len)

        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": clean_url,
                "source_type": "url",
                "scope": scope,
                "status": "EMBEDDING",
                "progress": 65,
                "message": f"Embedding {len(chunks)} chunks via CrispEmbed ({emb_model})...",
            }
        )
        vectors = await asyncio.to_thread(embedding_engine.embed_batch, chunks, emb_model)
        resolved_model = embedding_engine.model_name

        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": clean_url,
                "source_type": "url",
                "scope": scope,
                "status": "INDEXING",
                "progress": 85,
                "message": f"Indexing {len(chunks)} chunks into LanceDB hybrid store...",
            }
        )
        records = [
            {
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source_name": clean_url,
                "source_type": "url",
                "scope": scope,
                "session_id": session_id or (scope if scope != "global" else ""),
                "user_id": user_id,
                "embedding_model": resolved_model,
                "chunk_index": i,
                "text": chunk_str,
                "vector": vec,
            }
            for i, (chunk_str, vec) in enumerate(zip(chunks, vectors))
        ]
        await asyncio.to_thread(vector_store.add_chunks, records)

        updated_doc = store.update_document_status(
            doc_id, "SYNCED", chunk_count=len(chunks), file_size=byte_len
        )
        final_sse = _sse_event(
            {
                "doc_id": doc_id,
                "name": clean_url,
                "source_type": "url",
                "scope": scope,
                "status": "SYNCED",
                "progress": 100,
                "chunk_count": len(chunks),
                "document": updated_doc,
                "message": f"Synced {clean_url} ({len(chunks)} chunks)",
            }
        )
        if idempotency_key:
            store.save_idempotency(idempotency_key, final_sse, ttl_seconds=300.0)
        yield final_sse
    except Exception as exc:
        err_msg = str(exc)
        store.update_document_status(doc_id, "FAILED", error_message=err_msg)
        yield _sse_event(
            {
                "doc_id": doc_id,
                "name": clean_url,
                "source_type": "url",
                "scope": scope,
                "status": "FAILED",
                "progress": 100,
                "error": err_msg,
                "message": f"Failed to sync {clean_url}: {err_msg}",
            }
        )
