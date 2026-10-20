"""FastAPI application composition root."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.middleware import rate_limit_middleware
from app.api.routes import ROUTERS
from app.api.upload import read_upload_bounded as _read_upload_bounded
from app.bootstrap import seed_default_handbook_if_empty as _seed_default_handbook_if_empty
from app.config import BASE_DIR, MAX_UPLOAD_BYTES
from app.sample_data import SAMPLE_COMPANY_HANDBOOK

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _seed_default_handbook_if_empty()
    yield


app = FastAPI(
    title="Company RAG Chatbot — Swiss Monochrome Edition",
    version="1.0.0",
    lifespan=lifespan,
)

app.middleware("http")(rate_limit_middleware)

for router in ROUTERS:
    app.include_router(router)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_spa():
    return FileResponse(str(STATIC_DIR / "index.html"))


__all__ = [
    "MAX_UPLOAD_BYTES",
    "SAMPLE_COMPANY_HANDBOOK",
    "_read_upload_bounded",
    "_seed_default_handbook_if_empty",
    "app",
    "lifespan",
]
