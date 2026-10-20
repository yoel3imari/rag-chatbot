"""Session lifecycle and session-scoped document cleanup."""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional


def _remove_document_files(rows: List[Any]) -> None:
    for row in rows:
        if row["source_type"] == "file" and row["raw_path_or_url"]:
            try:
                if os.path.exists(row["raw_path_or_url"]):
                    os.remove(row["raw_path_or_url"])
            except OSError:
                pass


class SessionMixin:
    def list_sessions(self, user_id: str = "default") -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            result = []
            for row in rows:
                session = dict(row)
                session["linked_sessions"] = json.loads(session.get("linked_sessions") or "[]")
                result.append(session)
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
            session = dict(row)
            session["linked_sessions"] = json.loads(session.get("linked_sessions") or "[]")
            messages = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            session["messages"] = [
                {
                    **dict(message),
                    "react_steps": json.loads(message["react_steps"] or "[]"),
                    "citations": json.loads(message["citations"] or "[]"),
                    "included_in_context": bool(message["included_in_context"]),
                }
                for message in messages
            ]
            return session

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
        """Deletes a session and returns session-scoped document IDs to purge from LanceDB."""
        with self._lock, self._conn() as conn:
            document_rows = conn.execute(
                "SELECT id, raw_path_or_url, source_type FROM documents WHERE session_id = ? OR scope = ?",
                (session_id, session_id),
            ).fetchall()
            document_ids = [row["id"] for row in document_rows]
            _remove_document_files(document_rows)
            conn.execute(
                "DELETE FROM documents WHERE session_id = ? OR scope = ?",
                (session_id, session_id),
            )
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return document_ids

    def clear_all_sessions(self, user_id: str = "default") -> List[str]:
        """Clears all sessions and returns session-scoped document IDs to purge."""
        with self._lock, self._conn() as conn:
            document_rows = conn.execute(
                "SELECT id, raw_path_or_url, source_type FROM documents WHERE user_id = ? AND scope != 'global'",
                (user_id,),
            ).fetchall()
            document_ids = [row["id"] for row in document_rows]
            _remove_document_files(document_rows)
            conn.execute(
                "DELETE FROM documents WHERE user_id = ? AND scope != 'global'",
                (user_id,),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return document_ids
