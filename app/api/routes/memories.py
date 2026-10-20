"""Cross-session memory ledger routes."""

from fastapi import APIRouter, HTTPException

from app.models import MemoryCreate
from app.store import store

router = APIRouter(tags=["memories"])


@router.get("/api/memories")
async def list_memories():
    return store.list_memories()


@router.post("/api/memories")
async def add_memory(payload: MemoryCreate):
    if not payload.fact.strip():
        raise HTTPException(status_code=400, detail="Fact cannot be empty")
    return store.add_memory(
        fact=payload.fact, category=payload.category or "manual"
    )


@router.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    deleted = store.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "id": memory_id}
