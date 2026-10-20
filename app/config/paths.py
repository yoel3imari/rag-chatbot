"""Filesystem locations and runtime limits used across the application."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Tests (tests/conftest.py) redirect all runtime data into a temp dir via RAG_DATA_DIR.
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR") or (BASE_DIR / "data"))
LANCEDB_DIR = DATA_DIR / "lancedb"
UPLOADS_DIR = DATA_DIR / "uploads"
SQLITE_PATH = DATA_DIR / "app.db"
SECRET_KEY_PATH = DATA_DIR / ".secret_key"

MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_MB", "20")) * 1024 * 1024
EMBEDDING_DIM = 384

for _directory in (DATA_DIR, LANCEDB_DIR, UPLOADS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
