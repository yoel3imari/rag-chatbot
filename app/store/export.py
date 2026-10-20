"""Data export aggregation across store repositories."""

import time
from typing import Any, Dict


class ExportMixin:
    def export_all_history(self, user_id: str = "default") -> Dict[str, Any]:
        sessions = self.list_sessions(user_id=user_id)
        detailed_sessions = [self.get_session(session["id"]) for session in sessions]
        return {
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sessions": detailed_sessions,
            "memories": self.list_memories(user_id=user_id),
            "documents": self.list_documents(user_id=user_id),
            "eval_runs": self.list_eval_runs(user_id=user_id, limit=10),
        }
