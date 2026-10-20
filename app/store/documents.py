"""Document and web-source metadata records."""

import os
import time
import uuid
from typing import Any, Dict, List, Optional


class DocumentMixin:
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
        document_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, user_id, name, source_type, scope, session_id, file_size, chunk_count, status, error_message, raw_path_or_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'QUEUED', '', ?, ?, ?)
                """,
                (
                    document_id,
                    user_id,
                    name,
                    source_type,
                    scope,
                    session_id,
                    file_size,
                    raw_path_or_url,
                    now,
                    now,
                ),
            )
            return dict(
                conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            )

    def update_document_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        file_size: Optional[int] = None,
        error_message: str = "",
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
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
                (status, new_chunks, new_size, error_message, now, document_id),
            )
            return dict(
                conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            )

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
            return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def delete_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not row:
                return None
            document = dict(row)
            if document["source_type"] == "file" and document["raw_path_or_url"]:
                try:
                    if os.path.exists(document["raw_path_or_url"]):
                        os.remove(document["raw_path_or_url"])
                except OSError:
                    pass
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return document
