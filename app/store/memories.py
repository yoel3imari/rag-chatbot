"""Cross-session memory ledger."""

import time
import uuid
from typing import Any, Dict, List


class MemoryMixin:
    def list_memories(self, user_id: str = "default") -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_memory(
        self,
        fact: str,
        category: str = "manual",
        source_session_id: str = "",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        cleaned_fact = fact.strip()
        with self._lock, self._conn() as conn:
            existing = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND LOWER(fact) = LOWER(?)",
                (user_id, cleaned_fact),
            ).fetchone()
            if existing:
                return dict(existing)
            memory_id = str(uuid.uuid4())
            now = time.time()
            conn.execute(
                """
                INSERT INTO memories (id, user_id, fact, category, source_session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, user_id, cleaned_fact, category, source_session_id, now),
            )
            return dict(
                conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
            )

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock, self._conn() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0
