"""LangChain tools exposed to the ReAct agent."""

import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.vector_store import vector_store


class SearchKnowledgeBaseArgs(BaseModel):
    reasoning: str = Field(
        description="Brief ReAct thought explaining why retrieval is needed and what you are looking for."
    )
    query: str = Field(
        description="Optimized search query combining key semantic concepts and keywords."
    )
    target_source: Optional[str] = Field(
        default=None,
        description="Optional specific document filename or URL to filter by if mentioned.",
    )


@tool("search_knowledge_base", args_schema=SearchKnowledgeBaseArgs)
def search_knowledge_base(
    reasoning: str, query: str, target_source: Optional[str] = None
) -> str:
    """
    Search the company knowledge base (LanceDB hybrid vector + full-text search)
    for relevant document chunks and ingested web pages. Call this tool whenever
    the user asks about company policies, uploaded files, URLs, technical docs,
    or specific facts that may be stored in the knowledge base. Do NOT call this
    tool for simple greetings, chitchat, or pure math/general logic.

    NOTE: This bound-tool callable uses default scoping (global docs, "default"
    user) because the agent executes retrieval manually with the caller's
    user_id/session_id. If you ever wire this tool into a LangChain agent
    executor, thread the caller's user_id/session_id through first — otherwise
    document/session scoping is silently dropped.
    """
    targets = [target_source] if target_source else None
    results = vector_store.hybrid_search(query=query, target_docs=targets)
    return json.dumps(results)


RETRIEVAL_LC_TOOLS = [search_knowledge_base]
