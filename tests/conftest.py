"""
Isolates all tests from the developer's real `data/` directory.

Must be imported before any `app.*` module: app.config freezes DATA_DIR at
import time, and pytest always imports conftest before collecting test modules.
Setting RAG_DATA_DIR here redirects app.config, app.store, and app.vector_store
singleton paths into a per-run temp dir — lifespan seeding, document uploads,
and idempotency tests therefore never touch the real store, and repeated test
runs stay deterministic (idempotency caches start empty).
"""

import atexit
import os
import shutil
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="rag_chatbot_test_data_")
os.environ["RAG_DATA_DIR"] = _TEST_DATA_DIR
atexit.register(shutil.rmtree, _TEST_DATA_DIR, ignore_errors=True)
