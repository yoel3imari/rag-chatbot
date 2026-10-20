"""ReAct agent, re-exported for backwards-compatible `app.agent` imports."""

from app.agent.context import (
    _truncate_to_token_budget,
    assemble_tiered_context,
    maybe_compact_session_summary,
)
from app.agent.fallback import _synthesize_local_fallback_response
from app.agent.heuristics import (
    _extract_candidate_memory_facts,
    _is_fatal_upstream_error,
    _should_auto_retrieve_heuristic,
)
from app.agent.messages import to_langchain_messages
from app.agent.retrieval import _format_rag_context_block, _safe_hybrid_search
from app.agent.streaming import run_react_chat_stream
from app.agent.tools import (
    RETRIEVAL_LC_TOOLS,
    SearchKnowledgeBaseArgs,
    search_knowledge_base,
)

__all__ = [
    "RETRIEVAL_LC_TOOLS",
    "SearchKnowledgeBaseArgs",
    "_extract_candidate_memory_facts",
    "_format_rag_context_block",
    "_is_fatal_upstream_error",
    "_safe_hybrid_search",
    "_should_auto_retrieve_heuristic",
    "_synthesize_local_fallback_response",
    "_truncate_to_token_budget",
    "assemble_tiered_context",
    "maybe_compact_session_summary",
    "run_react_chat_stream",
    "search_knowledge_base",
    "to_langchain_messages",
]
