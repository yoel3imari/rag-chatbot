"""Streaming ingestion pipelines for files and URLs."""

import asyncio
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.config import UPLOADS_DIR
from app.embeddings import embedding_engine
from app.ingestion.chunking import chunk_text
from app.ingestion.extractors import clean_html_to_text, extract_text_from_file
from app.ingestion.fetcher import fetch_url_safely
from app.ingestion.sse import sse_event
from app.store import store
from app.vector_store import vector_store


def _document_event(
    doc_id: str,
    name: str,
    source_type: str,
    scope: str,
    status: str,
    progress: int,
    message: str,
    **extra: Any,
) -> str:
    return sse_event(
        {
            "doc_id": doc_id,
            "name": name,
            "source_type": source_type,
            "scope": scope,
            "status": status,
            "progress": progress,
            "message": message,
            **extra,
        }
    )


async def _embed_and_index_stream(
    *,
    doc_id: str,
    source_name: str,
    source_type: str,
    scope: str,
    session_id: str,
    user_id: str,
    chunks: List[str],
    final_status: str,
    indexing_message: str,
    final_message: str,
    idempotency_key: Optional[str],
    file_size: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Embeds parsed chunks and writes them to LanceDB, emitting progress events."""
    settings = store.get_settings(user_id=user_id)
    embedding_model = settings.get("embedding_model", "all-MiniLM-L6-v2")

    store.update_document_status(
        doc_id, "PROCESSING", chunk_count=len(chunks), file_size=file_size
    )
    yield _document_event(
        doc_id,
        source_name,
        source_type,
        scope,
        "EMBEDDING",
        65,
        f"Embedding {len(chunks)} chunks via CrispEmbed ({embedding_model})...",
    )

    vectors = await asyncio.to_thread(embedding_engine.embed_batch, chunks, embedding_model)
    resolved_model = embedding_engine.model_name

    yield _document_event(
        doc_id,
        source_name,
        source_type,
        scope,
        "INDEXING",
        85,
        indexing_message,
    )

    records = [
        {
            "id": f"{doc_id}_{index}",
            "doc_id": doc_id,
            "source_name": source_name,
            "source_type": source_type,
            "scope": scope,
            "session_id": session_id or (scope if scope != "global" else ""),
            "user_id": user_id,
            "embedding_model": resolved_model,
            "chunk_index": index,
            "text": chunk,
            "vector": vector,
        }
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    await asyncio.to_thread(vector_store.add_chunks, records)

    updated_document = store.update_document_status(
        doc_id, final_status, chunk_count=len(chunks), file_size=file_size
    )
    final_event = _document_event(
        doc_id,
        source_name,
        source_type,
        scope,
        final_status,
        100,
        final_message,
        chunk_count=len(chunks),
        document=updated_document,
    )
    if idempotency_key:
        store.save_idempotency(idempotency_key, final_event, ttl_seconds=300.0)
    yield final_event


async def ingest_file_stream(
    filename: str,
    file_bytes: bytes,
    scope: str = "global",
    session_id: str = "",
    user_id: str = "default",
    idempotency_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streams SSE ingestion events while parsing, embedding, and indexing a file."""
    safe_name = Path(filename).name
    file_size = len(file_bytes)
    save_path = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    save_path.write_bytes(file_bytes)

    document = store.create_document_record(
        name=safe_name,
        source_type="file",
        scope=scope,
        session_id=session_id,
        file_size=file_size,
        raw_path_or_url=str(save_path),
        user_id=user_id,
    )
    doc_id = document["id"]

    yield _document_event(
        doc_id,
        safe_name,
        "file",
        scope,
        "QUEUED",
        10,
        f"Queued {safe_name} ({file_size // 1024 or 1} KB)",
    )
    await asyncio.sleep(0.02)

    try:
        store.update_document_status(doc_id, "PARSING")
        yield _document_event(
            doc_id,
            safe_name,
            "file",
            scope,
            "PARSING",
            30,
            f"Parsing text from {safe_name}...",
        )
        text = await asyncio.to_thread(extract_text_from_file, file_bytes, safe_name)
        if not text.strip():
            raise ValueError("No extractable text found in document.")

        chunks = await asyncio.to_thread(chunk_text, text)
        if not chunks:
            raise ValueError("Document produced 0 text chunks.")

        async for event in _embed_and_index_stream(
            doc_id=doc_id,
            source_name=safe_name,
            source_type="file",
            scope=scope,
            session_id=session_id,
            user_id=user_id,
            chunks=chunks,
            final_status="INDEXED",
            indexing_message=f"Indexing {len(chunks)} vectors + FTS into LanceDB...",
            final_message=f"Indexed {safe_name} ({len(chunks)} chunks)",
            idempotency_key=idempotency_key,
        ):
            yield event
    except Exception as exc:
        message = str(exc)
        store.update_document_status(doc_id, "FAILED", error_message=message)
        yield _document_event(
            doc_id,
            safe_name,
            "file",
            scope,
            "FAILED",
            100,
            f"Failed to ingest {safe_name}: {message}",
            error=message,
        )


async def ingest_url_stream(
    url: str,
    scope: str = "global",
    session_id: str = "",
    user_id: str = "default",
    existing_doc_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streams SSE events while scraping, cleaning, embedding, and indexing a URL."""
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    if existing_doc_id:
        document = store.get_document(existing_doc_id)
        if not document:
            document = store.create_document_record(
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
        document = store.create_document_record(
            name=clean_url,
            source_type="url",
            scope=scope,
            session_id=session_id,
            raw_path_or_url=clean_url,
            user_id=user_id,
        )

    doc_id = document["id"]

    yield _document_event(
        doc_id,
        clean_url,
        "url",
        scope,
        "QUEUED",
        10,
        f"Validating & fetching {clean_url}...",
    )

    try:
        store.update_document_status(doc_id, "PARSING")
        yield _document_event(
            doc_id,
            clean_url,
            "url",
            scope,
            "PARSING",
            30,
            f"Scraping & stripping HTML boilerplate from {clean_url}...",
        )

        raw_html, byte_length = await fetch_url_safely(clean_url)
        text = await asyncio.to_thread(clean_html_to_text, raw_html, clean_url)
        if not text.strip():
            raise ValueError("No readable text content found at URL.")

        chunks = await asyncio.to_thread(chunk_text, text)
        if not chunks:
            raise ValueError("URL content produced 0 text chunks.")

        async for event in _embed_and_index_stream(
            doc_id=doc_id,
            source_name=clean_url,
            source_type="url",
            scope=scope,
            session_id=session_id,
            user_id=user_id,
            chunks=chunks,
            final_status="SYNCED",
            indexing_message=f"Indexing {len(chunks)} chunks into LanceDB hybrid store...",
            final_message=f"Synced {clean_url} ({len(chunks)} chunks)",
            idempotency_key=idempotency_key,
            file_size=byte_length,
        ):
            yield event
    except Exception as exc:
        message = str(exc)
        store.update_document_status(doc_id, "FAILED", error_message=message)
        yield _document_event(
            doc_id,
            clean_url,
            "url",
            scope,
            "FAILED",
            100,
            f"Failed to sync {clean_url}: {message}",
            error=message,
        )
