from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: Optional[str] = "New Session"


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None


class MessageToggleContext(BaseModel):
    included_in_context: bool


class LinkSessionRequest(BaseModel):
    target_session_id: str


class SettingsUpdate(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    dedup_threshold: Optional[float] = None
    theme: Optional[str] = None
    compaction_mode: Optional[str] = None
    max_context_tokens: Optional[int] = None
    reserved_output_tokens: Optional[int] = None
    rag_token_ceiling: Optional[int] = None
    sliding_window_turns: Optional[int] = None


class MemoryCreate(BaseModel):
    fact: str
    category: Optional[str] = "manual"


class UrlIngestRequest(BaseModel):
    url: str
    scope: str = "global"  # "global" or session_id
    session_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    target_docs: List[str] = Field(default_factory=list)  # list of doc_ids or source_names from @mention
    use_global_docs: bool = True
    use_memory: bool = True
    linked_session_ids: List[str] = Field(default_factory=list)
    idempotency_key: Optional[str] = None
    client_token_estimate: Optional[int] = None
