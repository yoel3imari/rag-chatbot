"""Metadata and settings routes."""

import asyncio

from fastapi import APIRouter

from app.config import EMBEDDING_MODELS, PROVIDER_PRESETS
from app.embeddings import embedding_engine
from app.models import SettingsUpdate
from app.store import store
from app.vector_store import vector_store

router = APIRouter(tags=["settings"])


@router.get("/api/meta")
async def get_meta():
    return {
        "providers": PROVIDER_PRESETS,
        "embedding_models": EMBEDDING_MODELS,
        "active_embedding_model": embedding_engine.model_name,
        "embedding_dim": embedding_engine.dim,
    }


@router.get("/api/settings")
async def get_settings():
    return store.get_settings(include_decrypted_key=False)


@router.put("/api/settings")
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
