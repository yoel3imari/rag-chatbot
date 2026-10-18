import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.config import (
    DEFAULT_SETTINGS,
    PROVIDER_ENV_KEY_MAP,
    SQLITE_PATH,
    decrypt_secret,
    encrypt_secret,
    mask_secret,
)


def estimate_tokens(text: str) -> int:
    """Lightweight heuristic token estimator (~3.8 chars/token + 1.3 words adjustment, aligned with JS worker)."""
    if not text:
        return 0
    trimmed = text.strip()
    if not trimmed:
        return 0
    words = len(trimmed.split())
    chars = len(trimmed)
    return max(1, round((chars / 3.8 + words * 1.3) / 2))


class RateLimiter:
    """In-memory sliding-window rate limiter per client key with periodic stale-bucket sweeping."""

    def __init__(self, max_requests_per_minute: int = 90):
        self.max_rpm = max_requests_per_minute
        self._lock = threading.Lock()
        self._buckets: Dict[str, List[float]] = {}
        self._last_sweep: float = time.time()

    def _sweep_stale_unlocked(self, window_start: float, now: float) -> None:
        if now - self._last_sweep < 60.0 and len(self._buckets) < 500:
            return
        stale_keys = []
        for k, ts_list in self._buckets.items():
            kept = [t for t in ts_list if t > window_start]
            if kept:
                self._buckets[k] = kept
            else:
                stale_keys.append(k)
        for k in stale_keys:
            self._buckets.pop(k, None)
        self._last_sweep = now

    def check(self, client_key: str) -> Tuple[bool, int]:
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            self._sweep_stale_unlocked(window_start, now)
            timestamps = [t for t in self._buckets.get(client_key, []) if t > window_start]
            if len(timestamps) >= self.max_rpm:
                retry_after = max(1, int(60.0 - (now - timestamps[0])))
                self._buckets[client_key] = timestamps
                return False, retry_after
            timestamps.append(now)
            self._buckets[client_key] = timestamps
            return True, 0


class AppStore:
    """Partitioned relational store (SQLite) + in-memory cache for sessions, documents, memory, and settings."""

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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    linked_sessions TEXT NOT NULL DEFAULT '[]',
                    token_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    react_steps TEXT NOT NULL DEFAULT '[]',
                    citations TEXT NOT NULL DEFAULT '[]',
                    tokens INTEGER NOT NULL DEFAULT 0,
                    included_in_context INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'global',
                    session_id TEXT NOT NULL DEFAULT '',
                    file_size INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'QUEUED',
                    error_message TEXT NOT NULL DEFAULT '',
                    raw_path_or_url TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    fact TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'manual',
                    source_session_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    settings_json TEXT NOT NULL,
                    encrypted_keys_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    mode TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    llm_model TEXT NOT NULL DEFAULT '',
                    overall_score REAL NOT NULL DEFAULT 0.0,
                    total_cases INTEGER NOT NULL DEFAULT 0,
                    passed_cases INTEGER NOT NULL DEFAULT 0,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    results_json TEXT NOT NULL DEFAULT '[]',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                """
            )

    # --- Idempotency ---
    def _sweep_idempotency_unlocked(self, now: float) -> None:
        expired = [k for k, (exp_ts, _) in self._idempotency_cache.items() if now >= exp_ts]
        for k in expired:
            self._idempotency_cache.pop(k, None)

    def check_idempotency(self, key: Optional[str]) -> Optional[Any]:
        if not key:
            return None
        now = time.time()
        with self._lock:
            self._sweep_idempotency_unlocked(now)
            if key in self._idempotency_cache:
                return self._idempotency_cache[key][1]
            return None

    def save_idempotency(self, key: Optional[str], result: Any, ttl_seconds: float = 300.0) -> None:
        if not key:
            return
        now = time.time()
        with self._lock:
            self._sweep_idempotency_unlocked(now)
            self._idempotency_cache[key] = (now + ttl_seconds, result)

    def release_idempotency(self, key: Optional[str]) -> None:
        if not key:
            return
        with self._lock:
            self._idempotency_cache.pop(key, None)

    # --- Settings & Provider-Isolated Encrypted Credentials ---
    def get_settings(self, user_id: str = "default", include_decrypted_key: bool = False) -> Dict[str, Any]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT settings_json, encrypted_keys_json FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            settings = dict(DEFAULT_SETTINGS)
            encrypted_keys: Dict[str, str] = {}
            if row:
                try:
                    saved = json.loads(row["settings_json"])
                    settings.update(saved)
                except Exception:
                    pass
                try:
                    encrypted_keys = json.loads(row["encrypted_keys_json"])
                except Exception:
                    pass

            # Strictly isolate keys by provider ID — never alias across providers
            provider = settings.get("provider", "openrouter")
            enc_key = encrypted_keys.get(provider, "")
            raw_key = decrypt_secret(enc_key) if enc_key else ""
            if not raw_key:
                env_var = PROVIDER_ENV_KEY_MAP.get(provider)
                if env_var:
                    raw_key = os.environ.get(env_var, "")

            settings["api_key_masked"] = mask_secret(raw_key)
            settings["has_api_key"] = bool(raw_key)

            provider_key_status = {}
            for p_id, e_val in encrypted_keys.items():
                if p_id.startswith("_"):
                    continue
                dec = decrypt_secret(e_val)
                if dec:
                    provider_key_status[p_id] = mask_secret(dec)
            settings["provider_keys_masked"] = provider_key_status

            if include_decrypted_key:
                settings["_decrypted_api_key"] = raw_key
            return settings

    def update_settings(self, updates: Dict[str, Any], user_id: str = "default") -> Dict[str, Any]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT settings_json, encrypted_keys_json FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            current = dict(DEFAULT_SETTINGS)
            encrypted_keys: Dict[str, str] = {}
            if row:
                try:
                    current.update(json.loads(row["settings_json"]))
                except Exception:
                    pass
                try:
                    encrypted_keys = json.loads(row["encrypted_keys_json"])
                except Exception:
                    pass

            # Remove any legacy _default key if present
            encrypted_keys.pop("_default", None)

            new_api_key = updates.pop("api_key", None)
            for k, v in updates.items():
                if v is not None and k in DEFAULT_SETTINGS:
                    current[k] = v

            provider = current.get("provider", "openrouter")
            if new_api_key is not None:
                cleaned = new_api_key.strip()
                if "••••" not in cleaned:
                    if cleaned == "":
                        encrypted_keys.pop(provider, None)
                    else:
                        encrypted_keys[provider] = encrypt_secret(cleaned)

            now = time.time()
            conn.execute(
                """
                INSERT INTO user_settings (user_id, settings_json, encrypted_keys_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    encrypted_keys_json = excluded.encrypted_keys_json,
                    updated_at = excluded.updated_at
                """,
                (user_id, json.dumps(current), json.dumps(encrypted_keys), now),
            )
        return self.get_settings(user_id=user_id, include_decrypted_key=False)

    # --- Sessions ---
    def list_sessions(self, user_id: str = "default") -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["linked_sessions"] = json.loads(d.get("linked_sessions") or "[]")
                result.append(d)
            return result

    def create_session(self, title: str = "New Session", user_id: str = "default") -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, user_id, title, summary, linked_sessions, token_count, created_at, updated_at)
                VALUES (?, ?, ?, '', '[]', 0, ?, ?)
                """,
                (session_id, user_id, title, now, now),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["linked_sessions"] = json.loads(d.get("linked_sessions") or "[]")
            msgs = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            d["messages"] = [
                {
                    **dict(m),
                    "react_steps": json.loads(m["react_steps"] or "[]"),
                    "citations": json.loads(m["citations"] or "[]"),
                    "included_in_context": bool(m["included_in_context"]),
                }
                for m in msgs
            ]
            return d

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        linked_sessions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return None
            new_title = title if title is not None else row["title"]
            new_summary = summary if summary is not None else row["summary"]
            new_linked = (
                json.dumps(linked_sessions)
                if linked_sessions is not None
                else row["linked_sessions"]
            )
            now = time.time()
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, summary = ?, linked_sessions = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_title, new_summary, new_linked, now, session_id),
            )
            self._recalculate_session_tokens(conn, session_id, now)
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> List[str]:
        """Deletes a session and returns any session-scoped document IDs to purge from LanceDB."""
        with self._lock, self._conn() as conn:
            doc_rows = conn.execute(
                "SELECT id, raw_path_or_url, source_type FROM documents WHERE session_id = ? OR scope = ?",
                (session_id, session_id),
            ).fetchall()
            doc_ids = [r["id"] for r in doc_rows]
            for r in doc_rows:
                if r["source_type"] == "file" and r["raw_path_or_url"]:
                    try:
                        if os.path.exists(r["raw_path_or_url"]):
                            os.remove(r["raw_path_or_url"])
                    except OSError:
                        pass
            conn.execute("DELETE FROM documents WHERE session_id = ? OR scope = ?", (session_id, session_id))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return doc_ids

    def clear_all_sessions(self, user_id: str = "default") -> List[str]:
        """Clears all sessions and returns session-scoped document IDs to purge."""
        with self._lock, self._conn() as conn:
            doc_rows = conn.execute(
                "SELECT id, raw_path_or_url, source_type FROM documents WHERE user_id = ? AND scope != 'global'",
                (user_id,),
            ).fetchall()
            doc_ids = [r["id"] for r in doc_rows]
            for r in doc_rows:
                if r["source_type"] == "file" and r["raw_path_or_url"]:
                    try:
                        if os.path.exists(r["raw_path_or_url"]):
                            os.remove(r["raw_path_or_url"])
                    except OSError:
                        pass
            conn.execute("DELETE FROM documents WHERE user_id = ? AND scope != 'global'", (user_id,))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return doc_ids

    # --- Messages ---
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        react_steps: Optional[List[Dict[str, Any]]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        included_in_context: bool = True,
    ) -> Dict[str, Any]:
        msg_id = str(uuid.uuid4())
        now = time.time()
        tokens = estimate_tokens(content)
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, react_steps, citations, tokens, included_in_context, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    session_id,
                    role,
                    content,
                    json.dumps(react_steps or []),
                    json.dumps(citations or []),
                    tokens,
                    1 if included_in_context else 0,
                    now,
                ),
            )
            self._recalculate_session_tokens(conn, session_id, now)
            return {
                "id": msg_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "react_steps": react_steps or [],
                "citations": citations or [],
                "tokens": tokens,
                "included_in_context": included_in_context,
                "created_at": now,
            }

    def toggle_message_context(self, message_id: str, included_in_context: bool) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE messages SET included_in_context = ? WHERE id = ?",
                (1 if included_in_context else 0, message_id),
            )
            self._recalculate_session_tokens(conn, row["session_id"], time.time())
            updated = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
            return {
                **dict(updated),
                "react_steps": json.loads(updated["react_steps"] or "[]"),
                "citations": json.loads(updated["citations"] or "[]"),
                "included_in_context": bool(updated["included_in_context"]),
            }

    def _recalculate_session_tokens(self, conn: sqlite3.Connection, session_id: str, now: float) -> None:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) AS total FROM messages WHERE session_id = ? AND included_in_context = 1",
            (session_id,),
        ).fetchone()
        sess = conn.execute("SELECT summary FROM sessions WHERE id = ?", (session_id,)).fetchone()
        summary_tokens = estimate_tokens(sess["summary"]) if sess and sess["summary"] else 0
        total = (row["total"] if row else 0) + summary_tokens
        conn.execute(
            "UPDATE sessions SET token_count = ?, updated_at = ? WHERE id = ?",
            (total, now, session_id),
        )

    # --- Documents & Web Sources ---
    def create_document_record(
        self,
        name: str,
        source_type: str,
        scope: str = "global",
        session_id: str = "",
        file_size: int = 0,
        raw_path_or_url: str = "",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, user_id, name, source_type, scope, session_id, file_size, chunk_count, status, error_message, raw_path_or_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'QUEUED', '', ?, ?, ?)
                """,
                (doc_id, user_id, name, source_type, scope, session_id, file_size, raw_path_or_url, now, now),
            )
            return dict(conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone())

    def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        file_size: Optional[int] = None,
        error_message: str = "",
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if not row:
                return None
            new_chunks = chunk_count if chunk_count is not None else row["chunk_count"]
            new_size = file_size if file_size is not None else row["file_size"]
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, file_size = ?, error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, new_chunks, new_size, error_message, now, doc_id),
            )
            return dict(conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone())

    def list_documents(
        self,
        user_id: str = "default",
        session_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            query = "SELECT * FROM documents WHERE user_id = ?"
            params: List[Any] = [user_id]
            if session_id:
                query += " AND (scope = 'global' OR scope = ? OR session_id = ?)"
                params.extend([session_id, session_id])
            if source_type:
                query += " AND source_type = ?"
                params.append(source_type)
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            if d["source_type"] == "file" and d["raw_path_or_url"]:
                try:
                    if os.path.exists(d["raw_path_or_url"]):
                        os.remove(d["raw_path_or_url"])
                except OSError:
                    pass
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return d

    # --- Global Memory Ledger ---
    def list_memories(self, user_id: str = "default") -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def add_memory(
        self,
        fact: str,
        category: str = "manual",
        source_session_id: str = "",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        fact_clean = fact.strip()
        with self._lock, self._conn() as conn:
            existing = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND LOWER(fact) = LOWER(?)",
                (user_id, fact_clean),
            ).fetchone()
            if existing:
                return dict(existing)
            mem_id = str(uuid.uuid4())
            now = time.time()
            conn.execute(
                """
                INSERT INTO memories (id, user_id, fact, category, source_session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mem_id, user_id, fact_clean, category, source_session_id, now),
            )
            return dict(conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone())

    def delete_memory(self, mem_id: str) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            return cur.rowcount > 0

    # --- Evaluation Runs ---
    def save_eval_run(self, run_data: Dict[str, Any], user_id: str = "default") -> Dict[str, Any]:
        run_id = run_data.get("id") or str(uuid.uuid4())
        now = float(run_data.get("created_at") or time.time())
        mode = str(run_data.get("mode", "full"))
        dataset_name = str(run_data.get("dataset_name", "Golden Benchmark"))
        embedding_model = str(run_data.get("embedding_model", "all-MiniLM-L6-v2"))
        llm_model = str(run_data.get("llm_model", ""))
        overall_score = float(run_data.get("overall_score", 0.0))
        total_cases = int(run_data.get("total_cases", 0))
        passed_cases = int(run_data.get("passed_cases", 0))
        summary = run_data.get("summary") or {}
        results = run_data.get("results") or []
        config = run_data.get("config") or {}

        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO eval_runs (
                    id, user_id, mode, dataset_name, embedding_model, llm_model,
                    overall_score, total_cases, passed_cases, summary_json,
                    results_json, config_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    user_id,
                    mode,
                    dataset_name,
                    embedding_model,
                    llm_model,
                    overall_score,
                    total_cases,
                    passed_cases,
                    json.dumps(summary),
                    json.dumps(results),
                    json.dumps(config),
                    now,
                ),
            )
        return self.get_eval_run(run_id) or {}

    def list_eval_runs(self, user_id: str = "default", limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["summary"] = json.loads(d.pop("summary_json", "{}") or "{}")
                d["results"] = json.loads(d.pop("results_json", "[]") or "[]")
                d["config"] = json.loads(d.pop("config_json", "{}") or "{}")
                out.append(d)
            return out

    def get_eval_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["summary"] = json.loads(d.pop("summary_json", "{}") or "{}")
            d["results"] = json.loads(d.pop("results_json", "[]") or "[]")
            d["config"] = json.loads(d.pop("config_json", "{}") or "{}")
            return d

    def delete_eval_run(self, run_id: str) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))
            return cur.rowcount > 0

    def export_all_history(self, user_id: str = "default") -> Dict[str, Any]:
        sessions = self.list_sessions(user_id=user_id)
        detailed_sessions = [self.get_session(s["id"]) for s in sessions]
        memories = self.list_memories(user_id=user_id)
        documents = self.list_documents(user_id=user_id)
        eval_runs = self.list_eval_runs(user_id=user_id, limit=10)
        return {
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sessions": detailed_sessions,
            "memories": memories,
            "documents": documents,
            "eval_runs": eval_runs,
        }


store = AppStore()
rate_limiter = RateLimiter(max_requests_per_minute=90)
