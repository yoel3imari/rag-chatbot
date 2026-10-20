"""HTTP routers."""

from app.api.routes import (
    chat,
    documents,
    evaluation,
    export,
    memories,
    sessions,
    settings,
)

ROUTERS = [
    settings.router,
    sessions.router,
    documents.router,
    memories.router,
    export.router,
    evaluation.router,
    chat.router,
]

__all__ = ["ROUTERS", "chat", "documents", "evaluation", "export", "memories", "sessions", "settings"]
