"""SQLite connection lifecycle shared by all store repositories."""

import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Tuple

from app.config import SQLITE_PATH
from app.store.rate_limiter import RateLimiter
from app.store.schema import SCHEMA_SQL


class StoreBase:
    """Owns the single SQLite connection and per-instance in-memory caches."""

    def __init__(self, db_path: str = str(SQLITE_PATH)):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._idempotency_cache: Dict[str, Tuple[float, Any]] = {}
        self._db_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Returns a single persistent SQLite connection guarded by self._lock."""
        if self._db_conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._db_conn = conn
        return self._db_conn

    def close(self) -> None:
        with self._lock:
            if self._db_conn is not None:
                try:
                    self._db_conn.close()
                except Exception:
                    pass
                self._db_conn = None

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(SCHEMA_SQL)


rate_limiter = RateLimiter(max_requests_per_minute=90)
