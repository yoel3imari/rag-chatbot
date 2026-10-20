"""Message persistence and context-pruning flags."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from app.store.tokens import estimate_tokens


class MessageMixin:
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        react_steps: Optional[List[Dict[str, Any]]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        included_in_context: bool = True,
    ) -> Dict[str, Any]:
        message_id = str(uuid.uuid4())
        now = time.time()
        tokens = estimate_tokens(content)
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, react_steps, citations, tokens, included_in_context, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
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
                "id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "react_steps": react_steps or [],
                "citations": citations or [],
                "tokens": tokens,
                "included_in_context": included_in_context,
                "created_at": now,
            }

    def toggle_message_context(
        self, message_id: str, included_in_context: bool
    ) -> Optional[Dict[str, Any]]:
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

    def _recalculate_session_tokens(self, conn, session_id: str, now: float) -> None:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) AS total FROM messages WHERE session_id = ? AND included_in_context = 1",
            (session_id,),
        ).fetchone()
        session = conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        summary_tokens = estimate_tokens(session["summary"]) if session and session["summary"] else 0
        total = (row["total"] if row else 0) + summary_tokens
        conn.execute(
            "UPDATE sessions SET token_count = ?, updated_at = ? WHERE id = ?",
            (total, now, session_id),
        )
