"""ReAct chat streaming route."""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agent import run_react_chat_stream
from app.models import ChatRequest
from app.store import store

router = APIRouter(tags=["chat"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/api/chat/stream")
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
        store.save_idempotency(
            payload.idempotency_key, "in_flight", ttl_seconds=60.0
        )

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
        headers=SSE_HEADERS,
    )
