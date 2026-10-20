"""Session and message-context routes."""

import asyncio

from fastapi import APIRouter, HTTPException

from app.models import (
    LinkSessionRequest,
    MessageToggleContext,
    SessionCreate,
    SessionUpdate,
)
from app.store import store
from app.vector_store import vector_store

router = APIRouter(tags=["sessions"])


@router.get("/api/sessions")
async def list_sessions():
    sessions = store.list_sessions()
    if not sessions:
        return [store.create_session(title="New Session")]
    return sessions


@router.post("/api/sessions")
async def create_session(payload: SessionCreate):
    return store.create_session(title=payload.title or "New Session")


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, payload: SessionUpdate):
    session = store.update_session(
        session_id=session_id,
        title=payload.title,
        summary=payload.summary,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/api/sessions/{session_id}/link")
async def link_session(session_id: str, payload: LinkSessionRequest):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    target_session = store.get_session(payload.target_session_id)
    if not target_session:
        raise HTTPException(
            status_code=404, detail="Target session to link not found"
        )

    linked = list(session.get("linked_sessions") or [])
    if (
        payload.target_session_id not in linked
        and payload.target_session_id != session_id
    ):
        linked.append(payload.target_session_id)
    return store.update_session(session_id=session_id, linked_sessions=linked)


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    purged_document_ids = store.delete_session(session_id)
    if purged_document_ids:
        await asyncio.to_thread(
            vector_store.delete_by_doc_ids, purged_document_ids
        )
    return {"deleted": True, "purged_docs": len(purged_document_ids)}


@router.delete("/api/sessions")
async def clear_all_sessions():
    purged_document_ids = store.clear_all_sessions()
    if purged_document_ids:
        await asyncio.to_thread(
            vector_store.delete_by_doc_ids, purged_document_ids
        )
    return {
        "cleared": True,
        "active_session": store.create_session(title="New Session"),
    }


@router.patch("/api/messages/{message_id}/context")
async def toggle_message_context(
    message_id: str, payload: MessageToggleContext
):
    updated = store.toggle_message_context(
        message_id, payload.included_in_context
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Message not found")
    return updated
