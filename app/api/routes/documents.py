"""Document upload, URL ingestion, and source-management routes."""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.upload import current_max_upload_bytes, read_upload_bounded
from app.ingestion import ingest_file_stream, ingest_url_stream
from app.models import UrlIngestRequest
from app.store import store
from app.vector_store import vector_store

router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _cached_stream(cached: str) -> StreamingResponse:
    return StreamingResponse(iter([cached]), media_type="text/event-stream")


@router.get("/api/documents")
async def list_documents(
    session_id: Optional[str] = None, source_type: Optional[str] = None
):
    return store.list_documents(session_id=session_id, source_type=source_type)


@router.post("/api/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    scope: str = Form("global"),
    session_id: str = Form(""),
    idempotency_key: Optional[str] = Form(None),
):
    if idempotency_key:
        cached = store.check_idempotency(idempotency_key)
        if cached:
            return _cached_stream(cached)

    filename = file.filename or "document.txt"
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: .PDF, .TXT, .MD",
        )

    file_bytes = await read_upload_bounded(
        file, max_bytes=current_max_upload_bytes()
    )
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    effective_scope = scope if scope == "global" else (session_id or scope)

    return StreamingResponse(
        ingest_file_stream(
            filename=filename,
            file_bytes=file_bytes,
            scope=effective_scope,
            session_id=session_id,
            idempotency_key=idempotency_key,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/api/ingest/url")
async def ingest_url(payload: UrlIngestRequest):
    if payload.idempotency_key:
        cached = store.check_idempotency(payload.idempotency_key)
        if cached:
            return _cached_stream(cached)

    effective_scope = (
        payload.scope
        if payload.scope == "global"
        else (payload.session_id or payload.scope)
    )
    return StreamingResponse(
        ingest_url_stream(
            url=payload.url,
            scope=effective_scope,
            session_id=payload.session_id or "",
            idempotency_key=payload.idempotency_key,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/api/documents/{doc_id}/resync")
async def resync_url_document(doc_id: str):
    document = store.get_document(doc_id)
    if not document or document["source_type"] != "url":
        raise HTTPException(status_code=404, detail="URL source not found")

    return StreamingResponse(
        ingest_url_stream(
            url=document["raw_path_or_url"] or document["name"],
            scope=document["scope"],
            session_id=document["session_id"],
            existing_doc_id=doc_id,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    deleted = store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    await asyncio.to_thread(vector_store.delete_by_doc_ids, [doc_id])
    return {"deleted": True, "id": doc_id}
