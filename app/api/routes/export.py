"""Data export route."""

from fastapi import APIRouter

from app.store import store

router = APIRouter(tags=["export"])


@router.get("/api/export")
async def export_data():
    return store.export_all_history()
