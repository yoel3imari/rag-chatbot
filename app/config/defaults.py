"""Default prompts and application settings."""

import os
from typing import Any, Dict

DEFAULT_SYSTEM_PROMPT = (
    "You are an enterprise RAG knowledge assistant operating under a ReAct "
    "(Reasoning + Acting) protocol. Answer accurately, concisely, and with high "
    "information density. When a user asks about company policies, uploaded documents, "
    "ingested URLs, or domain-specific facts, invoke the `search_knowledge_base` tool first. "
    "When the user engages in general chitchat, math, or pure logic that does not depend on "
    "internal documents, respond directly without calling retrieval tools. Always cite the "
    "source document or URL name in brackets (e.g., [policy.pdf]) when using retrieved context."
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "provider": os.environ.get("LLM_PROVIDER", "openrouter"),
    "base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    "model": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
    "temperature": 0.2,
    "max_tokens": 2048,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "similarity_threshold": 0.15,
    "dedup_threshold": 0.92,
    "theme": "dark",
    "compaction_mode": "Auto-Summarize",  # Sliding Window | Auto-Summarize | Strict Truncation
    "max_context_tokens": 8192,
    "reserved_output_tokens": 2048,
    "rag_token_ceiling": 1500,
    "sliding_window_turns": 10,
}
