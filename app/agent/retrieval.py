"""Hybrid retrieval helpers and grounded-context formatting."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from app.store import estimate_tokens
from app.vector_store import vector_store


async def _safe_hybrid_search(
    **search_kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Runs `vector_store.hybrid_search` off the event loop and converts any failure
    (embedding load error, LanceDB error, reindex failure) into an error detail
    string, so a retrieval failure can never abort the ongoing SSE stream.
    """
    try:
        chunks = await asyncio.to_thread(vector_store.hybrid_search, **search_kwargs)
        return chunks, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:160]}"


def _format_rag_context_block(
    chunks: List[Dict[str, Any]], rag_token_ceiling: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """Formats retrieved chunks within the hard RAG token ceiling."""
    if not chunks:
        return "No matching documents found in LanceDB knowledge base.", []

    formatted_parts: List[str] = []
    kept_citations: List[Dict[str, Any]] = []
    used_tokens = 0

    for index, chunk in enumerate(chunks):
        block = (
            f"[Source {index + 1}: {chunk['source_name']} | Chunk #{chunk['chunk_index']} | "
            f"Score: {chunk['hybrid_score']} ({chunk['match_type']})]\n{chunk['text']}"
        )
        block_tokens = estimate_tokens(block)
        if used_tokens + block_tokens > rag_token_ceiling and formatted_parts:
            break
        formatted_parts.append(block)
        used_tokens += block_tokens
        kept_citations.append(
            {
                "index": index + 1,
                "doc_id": chunk["doc_id"],
                "source_name": chunk["source_name"],
                "source_type": chunk["source_type"],
                "scope": chunk["scope"],
                "chunk_index": chunk["chunk_index"],
                "similarity": chunk["similarity"],
                "hybrid_score": chunk["hybrid_score"],
                "match_type": chunk["match_type"],
                "preview": chunk["text"][:240]
                + ("..." if len(chunk["text"]) > 240 else ""),
            }
        )

    return "\n\n---\n\n".join(formatted_parts), kept_citations
