"""Public `AppStore` facade composed from focused repository mixins."""

from app.store.base import StoreBase
from app.store.documents import DocumentMixin
from app.store.evaluations import EvaluationMixin
from app.store.export import ExportMixin
from app.store.idempotency import IdempotencyMixin
from app.store.memories import MemoryMixin
from app.store.messages import MessageMixin
from app.store.sessions import SessionMixin
from app.store.settings import SettingsMixin


class AppStore(
    SettingsMixin,
    SessionMixin,
    MessageMixin,
    DocumentMixin,
    MemoryMixin,
    EvaluationMixin,
    ExportMixin,
    IdempotencyMixin,
    StoreBase,
):
    """Partitioned relational store for sessions, documents, memory, and settings."""
