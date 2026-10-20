"""Persisted evaluation scorecards."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional


def _decode_eval_row(row: Any) -> Dict[str, Any]:
    data = dict(row)
    data["summary"] = json.loads(data.pop("summary_json", "{}") or "{}")
    data["results"] = json.loads(data.pop("results_json", "[]") or "[]")
    data["config"] = json.loads(data.pop("config_json", "{}") or "{}")
    return data


class EvaluationMixin:
    def save_eval_run(
        self, run_data: Dict[str, Any], user_id: str = "default"
    ) -> Dict[str, Any]:
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

    def list_eval_runs(
        self, user_id: str = "default", limit: int = 25
    ) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [_decode_eval_row(row) for row in rows]

    def get_eval_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return _decode_eval_row(row) if row else None

    def delete_eval_run(self, run_id: str) -> bool:
        with self._lock, self._conn() as conn:
            cursor = conn.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))
            return cursor.rowcount > 0
