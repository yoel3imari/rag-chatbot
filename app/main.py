import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import json
from app.agent import run_react_chat_stream
from app.config import BASE_DIR, EMBEDDING_MODELS, MAX_UPLOAD_BYTES, PROVIDER_PRESETS
from app.embeddings import embedding_engine
from app.evaluation import (
    GOLDEN_EVAL_DATASET,
    generate_synthetic_eval_cases,
    run_evaluation,
    run_evaluation_stream,
)
from app.ingestion import chunk_text, ingest_file_stream, ingest_url_stream
from app.models import (
    ChatRequest,
    EvalRunRequest,
    LinkSessionRequest,
    MemoryCreate,
    MessageToggleContext,
    SessionCreate,
    SessionUpdate,
    SettingsUpdate,
    SyntheticEvalRequest,
    UrlIngestRequest,
)
from app.store import rate_limiter, store
from app.vector_store import vector_store

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


SAMPLE_COMPANY_HANDBOOK = """# ACME CORP — ENGINEERING & OPERATIONS HANDBOOK (2026)

## 1. Remote Work & Core Hours
Acme Corp operates on a hybrid-first engineering model. Core synchronous collaboration hours are 13:00 UTC to 17:00 UTC Monday through Friday.
All full-time employees receive an annual $2,500 Home Office Stipend and $1,200 Learning & Conference budget.

## 2. Paid Time Off (PTO) & Sabbatical Policy
Employees accrue 25 days of Paid Time Off (PTO) per calendar year, plus 12 company holidays.
After 4 years of continuous tenure, engineers qualify for a 4-week paid sabbatical.

## 3. Production Deployment & Incident Response
- **Deployment Freeze:** No routine production deployments on Fridays after 15:00 UTC without VP of Engineering sign-off.
- **SEV-1 SLA:** Critical customer-facing outages require initial on-call acknowledgement via PagerDuty within 5 minutes and a public status page update within 15 minutes.
- **Postmortems:** Blameless postmortems must be published in the `/incidents` repository within 48 hours of resolution.

## 4. Security & Credential Hygiene
All API keys and production secrets must be encrypted at rest using AES-256-GCM or stored in Vault. Rotating service tokens every 90 days is mandatory.
"""


async def _read_upload_bounded(upload: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Stream-reads an UploadFile in 64KB chunks, aborting with HTTP 413 if max_bytes is exceeded."""
    chunks = bytearray()
    chunk_size = 64 * 1024
    while True:
        piece = await upload.read(chunk_size)
        if not piece:
            break
        chunks.extend(piece)
        if len(chunks) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MB.",
            )
    return bytes(chunks)


async def _seed_default_handbook_if_empty():
    docs = store.list_documents()
    if len(docs) == 0:
        doc = store.create_document_record(
            name="acme-handbook-2026.md",
            source_type="file",
            scope="global",
            file_size=len(SAMPLE_COMPANY_HANDBOOK.encode("utf-8")),
        )
        doc_id = doc["id"]
        chunks = chunk_text(SAMPLE_COMPANY_HANDBOOK)
        vectors = await asyncio.to_thread(embedding_engine.embed_batch, chunks)
        records = [
            {
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source_name": "acme-handbook-2026.md",
                "source_type": "file",
                "scope": "global",
                "session_id": "",
                "user_id": "default",
                "embedding_model": embedding_engine.model_name,
                "chunk_index": i,
                "text": c,
                "vector": v,
            }
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        await asyncio.to_thread(vector_store.add_chunks, records)
        store.update_document_status(doc_id, "INDEXED", chunk_count=len(chunks))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _seed_default_handbook_if_empty()
    yield


app = FastAPI(
    title="Company RAG Chatbot — Swiss Monochrome Edition",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "local"
        allowed, retry_after = rate_limiter.check(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Retry in {retry_after}s.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


# --- Metadata & Settings Endpoints ---
@app.get("/api/meta")
async def get_meta():
    return {
        "providers": PROVIDER_PRESETS,
        "embedding_models": EMBEDDING_MODELS,
        "active_embedding_model": embedding_engine.model_name,
        "embedding_dim": embedding_engine.dim,
    }


@app.get("/api/settings")
async def get_settings():
    return store.get_settings(include_decrypted_key=False)


@app.put("/api/settings")
async def update_settings(payload: SettingsUpdate):
    updates = payload.model_dump(exclude_none=True)
    reindexed_chunks = 0
    if "embedding_model" in updates:
        reindexed_chunks = await asyncio.to_thread(
            vector_store.reindex_for_model, updates["embedding_model"]
        )
    updated_settings = store.update_settings(updates)
    updated_settings["reindexed_chunks"] = reindexed_chunks
    return updated_settings


# --- Session Endpoints ---
@app.get("/api/sessions")
async def list_sessions():
    sessions = store.list_sessions()
    if not sessions:
        created = store.create_session(title="New Session")
        return [created]
    return sessions


@app.post("/api/sessions")
async def create_session(payload: SessionCreate):
    return store.create_session(title=payload.title or "New Session")


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    sess = store.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, payload: SessionUpdate):
    sess = store.update_session(
        session_id=session_id,
        title=payload.title,
        summary=payload.summary,
    )
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.post("/api/sessions/{session_id}/link")
async def link_session(session_id: str, payload: LinkSessionRequest):
    sess = store.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    target_sess = store.get_session(payload.target_session_id)
    if not target_sess:
        raise HTTPException(status_code=404, detail="Target session to link not found")
    linked = list(sess.get("linked_sessions") or [])
    if payload.target_session_id not in linked and payload.target_session_id != session_id:
        linked.append(payload.target_session_id)
    updated = store.update_session(session_id=session_id, linked_sessions=linked)
    return updated


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    purged_doc_ids = store.delete_session(session_id)
    if purged_doc_ids:
        await asyncio.to_thread(vector_store.delete_by_doc_ids, purged_doc_ids)
    return {"deleted": True, "purged_docs": len(purged_doc_ids)}


@app.delete("/api/sessions")
async def clear_all_sessions():
    purged_doc_ids = store.clear_all_sessions()
    if purged_doc_ids:
        await asyncio.to_thread(vector_store.delete_by_doc_ids, purged_doc_ids)
    fresh = store.create_session(title="New Session")
    return {"cleared": True, "active_session": fresh}


@app.patch("/api/messages/{message_id}/context")
async def toggle_message_context(message_id: str, payload: MessageToggleContext):
    updated = store.toggle_message_context(message_id, payload.included_in_context)
    if not updated:
        raise HTTPException(status_code=404, detail="Message not found")
    return updated


# --- Documents & Web Sources Endpoints ---
@app.get("/api/documents")
async def list_documents(session_id: Optional[str] = None, source_type: Optional[str] = None):
    return store.list_documents(session_id=session_id, source_type=source_type)


@app.post("/api/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    scope: str = Form("global"),
    session_id: str = Form(""),
    idempotency_key: Optional[str] = Form(None),
):
    if idempotency_key:
        cached = store.check_idempotency(idempotency_key)
        if cached:
            return StreamingResponse(iter([cached]), media_type="text/event-stream")

    allowed_exts = {".pdf", ".txt", ".md", ".markdown"}
    filename = file.filename or "document.txt"
    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .PDF, .TXT, .MD",
        )

    file_bytes = await _read_upload_bounded(file, max_bytes=MAX_UPLOAD_BYTES)
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ingest/url")
async def ingest_url(payload: UrlIngestRequest):
    if payload.idempotency_key:
        cached = store.check_idempotency(payload.idempotency_key)
        if cached:
            return StreamingResponse(iter([cached]), media_type="text/event-stream")

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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/documents/{doc_id}/resync")
async def resync_url_document(doc_id: str):
    doc = store.get_document(doc_id)
    if not doc or doc["source_type"] != "url":
        raise HTTPException(status_code=404, detail="URL source not found")

    return StreamingResponse(
        ingest_url_stream(
            url=doc["raw_path_or_url"] or doc["name"],
            scope=doc["scope"],
            session_id=doc["session_id"],
            existing_doc_id=doc_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    deleted = store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    await asyncio.to_thread(vector_store.delete_by_doc_ids, [doc_id])
    return {"deleted": True, "id": doc_id}


# --- Global Memory Ledger Endpoints ---
@app.get("/api/memories")
async def list_memories():
    return store.list_memories()


@app.post("/api/memories")
async def add_memory(payload: MemoryCreate):
    if not payload.fact.strip():
        raise HTTPException(status_code=400, detail="Fact cannot be empty")
    return store.add_memory(fact=payload.fact, category=payload.category or "manual")


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    ok = store.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "id": memory_id}


# --- Data Export ---
@app.get("/api/export")
async def export_data():
    return store.export_all_history()


# --- RAG & ReAct Evaluation Endpoints ---
@app.get("/api/eval/dataset")
async def get_default_eval_dataset():
    return {
        "name": "Golden Benchmark Suite",
        "cases": GOLDEN_EVAL_DATASET,
    }


@app.post("/api/eval/generate")
async def generate_synthetic_dataset(payload: SyntheticEvalRequest):
    cases = await generate_synthetic_eval_cases(
        doc_ids=payload.doc_ids,
        max_cases=max(1, min(30, payload.max_cases)),
        use_llm=payload.use_llm,
    )
    return {
        "name": f"Synthetic Corpus Suite ({len(cases)} cases)",
        "cases": cases,
    }


@app.get("/api/eval/runs")
async def list_evaluation_runs():
    return store.list_eval_runs()


@app.get("/api/eval/runs/{run_id}")
async def get_evaluation_run(run_id: str):
    run = store.get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


@app.delete("/api/eval/runs/{run_id}")
async def delete_evaluation_run(run_id: str):
    ok = store.delete_eval_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return {"deleted": True, "id": run_id}


@app.post("/api/eval/run")
async def trigger_evaluation_run(payload: EvalRunRequest):
    return await run_evaluation(
        mode=payload.mode,
        dataset_source=payload.dataset_source,
        custom_cases=payload.custom_cases,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
        dedup_threshold=payload.dedup_threshold,
        embedding_model=payload.embedding_model,
        max_synthetic_cases=payload.max_synthetic_cases,
        persist=True,
    )


@app.post("/api/eval/run/stream")
async def trigger_evaluation_stream(payload: EvalRunRequest):
    async def _sse_eval_gen():
        async for ev in run_evaluation_stream(
            mode=payload.mode,
            dataset_source=payload.dataset_source,
            custom_cases=payload.custom_cases,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
            dedup_threshold=payload.dedup_threshold,
            embedding_model=payload.embedding_model,
            max_synthetic_cases=payload.max_synthetic_cases,
            persist=True,
        ):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        _sse_eval_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Chat & ReAct Agent SSE Endpoint ---
@app.post("/api/chat/stream")
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    x_llm_api_key: Optional[str] = Header(default=None),
):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if payload.idempotency_key:
        if store.check_idempotency(payload.idempotency_key):
            raise HTTPException(
                status_code=409,
                detail="Duplicate request with identical idempotency_key is currently in flight.",
            )
        store.save_idempotency(payload.idempotency_key, "in_flight", ttl_seconds=60.0)

    return StreamingResponse(
        run_react_chat_stream(
            request=request,
            session_id=payload.session_id,
            user_message=payload.message,
            target_docs=payload.target_docs,
            use_global_docs=payload.use_global_docs,
            use_memory=payload.use_memory,
            linked_session_ids=payload.linked_session_ids,
            header_api_key=x_llm_api_key,
            idempotency_key=payload.idempotency_key,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Static SPA Mount ---
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_spa():
    return FileResponse(str(STATIC_DIR / "index.html"))
