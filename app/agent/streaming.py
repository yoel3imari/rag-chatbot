"""ReAct chat orchestration and Server-Sent Event streaming."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.context import assemble_tiered_context, maybe_compact_session_summary
from app.agent.fallback import _synthesize_local_fallback_response
from app.agent.heuristics import (
    _extract_candidate_memory_facts,
    _is_fatal_upstream_error,
    _should_auto_retrieve_heuristic,
)
from app.agent.messages import to_langchain_messages
from app.agent.retrieval import _format_rag_context_block, _safe_hybrid_search
from app.agent.tools import RETRIEVAL_LC_TOOLS
from app.config import KEYLESS_PROVIDERS
from app.embeddings import embedding_engine
from app.store import store


@dataclass
class ChatTurn:
    """Everything an individual chat turn needs while streaming."""

    request: Request
    session_id: str
    user_message: str
    target_docs: List[str]
    use_global_docs: bool
    use_memory: bool
    linked_session_ids: Optional[List[str]]
    user_id: str
    settings: Dict[str, Any]
    api_key: str
    provider: str
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    top_k: int
    similarity_threshold: float
    dedup_threshold: float
    rag_ceiling: int
    embedding_model: str
    context_messages: List[Dict[str, str]]
    react_steps: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    did_retrieve: bool = False

    @property
    def has_live_llm(self) -> bool:
        return bool(self.api_key) or self.provider in KEYLESS_PROVIDERS


def _sse(event_type: str, payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


def _record_step(turn: ChatTurn, step: str, **fields: Any) -> str:
    event = {"step": step, **fields}
    turn.react_steps.append(event)
    return _sse("react_step", event)


def _build_llm(turn: ChatTurn) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=turn.api_key or "not-needed",
        base_url=turn.base_url,
        model=turn.model_name,
        temperature=turn.temperature,
        max_tokens=turn.max_tokens,
        timeout=45.0,
        max_retries=2,
    )


def _build_user_prompt(turn: ChatTurn) -> str:
    if not turn.target_docs:
        return turn.user_message
    return (
        f"[Targeted Document Filter Active: {', '.join(turn.target_docs)}]\n"
        f"User Query: {turn.user_message}\n"
        "(Note: Please call `search_knowledge_base` to inspect the targeted document(s).)"
    )


async def _search(turn: ChatTurn, **overrides: Any) -> Any:
    search_kwargs = {
        "query": turn.user_message,
        "user_id": turn.user_id,
        "session_id": turn.session_id,
        "use_global_docs": turn.use_global_docs,
        "target_docs": turn.target_docs or None,
        "top_k": turn.top_k,
        "similarity_threshold": turn.similarity_threshold,
        "dedup_threshold": turn.dedup_threshold,
        "embedding_model": turn.embedding_model,
    }
    search_kwargs.update(overrides)
    return await _safe_hybrid_search(**search_kwargs)


def _observation_step(turn: ChatTurn) -> str:
    sources = ", ".join(
        dict.fromkeys(citation["source_name"] for citation in turn.citations)
    ) or "0 matches"
    return _record_step(
        turn,
        "OBSERVATION",
        chunks_count=len(turn.citations),
        detail=f"Retrieved {len(turn.citations)} hybrid chunks from LanceDB ({sources}).",
    )


async def _finish_turn(
    turn: ChatTurn,
    content: str,
    compact: bool = True,
) -> AsyncGenerator[str, None]:
    """Persists the assistant response, emits `done`, and optionally compacts the session."""
    assistant_message = store.add_message(
        session_id=turn.session_id,
        role="assistant",
        content=content,
        react_steps=turn.react_steps,
        citations=turn.citations,
    )
    yield _sse("done", {"message": assistant_message})

    if compact and not await turn.request.is_disconnected():
        new_summary = await maybe_compact_session_summary(
            turn.session_id, turn.settings, turn.api_key
        )
        if new_summary:
            yield _sse("summary_updated", {"summary": new_summary})


async def _stream_fatal_fallback(
    turn: ChatTurn, upstream_error: str
) -> AsyncGenerator[str, None]:
    yield _record_step(
        turn,
        "ERROR",
        detail=f"Upstream provider error ({turn.provider}): {upstream_error[:160]}",
    )

    should_retrieve, _ = _should_auto_retrieve_heuristic(
        turn.user_message, turn.target_docs
    )
    if should_retrieve:
        turn.did_retrieve = True
        turn.retrieved_chunks, retrieval_error = await _search(turn)
        if retrieval_error:
            yield _record_step(
                turn,
                "ERROR",
                detail=f"Knowledge base retrieval failed: {retrieval_error}",
            )
        _, turn.citations = _format_rag_context_block(
            turn.retrieved_chunks, turn.rag_ceiling
        )
        if turn.citations:
            yield _sse("citations", {"citations": turn.citations})

    response = _synthesize_local_fallback_response(
        user_message=turn.user_message,
        citations=turn.citations,
        chunks=turn.retrieved_chunks,
        did_retrieve=turn.did_retrieve,
        provider=turn.provider,
        model=turn.model_name,
        upstream_error=upstream_error,
    )
    yield _sse("token", {"delta": response})
    async for event in _finish_turn(turn, response, compact=False):
        yield event


async def _stream_direct_answer(
    turn: ChatTurn,
    llm: ChatOpenAI,
    lc_messages: List[Any],
    direct_ai_message: Optional[AIMessage],
) -> AsyncGenerator[str, None]:
    yield _record_step(
        turn,
        "THOUGHT",
        detail="Direct response selected (no knowledge base retrieval required).",
    )

    direct_text = str(direct_ai_message.content if direct_ai_message else "").strip()
    if not direct_text:
        # The model chose not to call a tool; stream a direct completion.
        try:
            async for chunk in llm.astream(lc_messages):
                if await turn.request.is_disconnected():
                    break
                piece = str(chunk.content or "")
                if piece:
                    direct_text += piece
                    yield _sse("token", {"delta": piece})
        except Exception as stream_error:
            suffix = (
                f"\n\n⚠️ **Upstream LLM Error ({turn.provider}):** `{str(stream_error)}`"
            )
            direct_text += suffix
            yield _sse("token", {"delta": suffix})
    else:
        chunk_size = max(6, len(direct_text) // 40)
        for index in range(0, len(direct_text), chunk_size):
            if await turn.request.is_disconnected():
                break
            piece = direct_text[index : index + chunk_size]
            yield _sse("token", {"delta": piece})
            await asyncio.sleep(0.01)

    async for event in _finish_turn(
        turn, direct_text or "[Empty model response]", compact=True
    ):
        yield event


def _deduplicate_retrieved_chunks(
    turn: ChatTurn, combined_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    seen_ids = set()
    unique_chunks = []
    for chunk in combined_chunks:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            unique_chunks.append(chunk)
    return embedding_engine.deduplicate_chunks(
        unique_chunks, threshold=turn.dedup_threshold
    )[: turn.top_k]


async def _stream_retrieval_answer(
    turn: ChatTurn, llm: ChatOpenAI, tool_invocations: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    turn.did_retrieve = True
    combined_chunks: List[Dict[str, Any]] = []

    for invocation in tool_invocations:
        effective_targets = list(turn.target_docs)
        target_source = invocation["target_source"]
        if target_source and target_source not in effective_targets:
            effective_targets.append(target_source)

        yield _record_step(turn, "THOUGHT", detail=invocation["reasoning"])

        query = invocation["query"]
        filter_description = (
            f" | @filter={effective_targets}" if effective_targets else ""
        )
        yield _record_step(
            turn,
            "ACTION",
            tool="search_knowledge_base",
            query=query,
            target_docs=effective_targets,
            detail=f'search_knowledge_base(query="{query}"{filter_description})',
        )

        call_chunks, retrieval_error = await _search(
            turn,
            query=query,
            target_docs=effective_targets or None,
        )
        if retrieval_error:
            yield _record_step(
                turn,
                "ERROR",
                detail=f"Knowledge base retrieval failed: {retrieval_error}",
            )
        else:
            combined_chunks.extend(call_chunks)

    turn.retrieved_chunks = _deduplicate_retrieved_chunks(turn, combined_chunks)
    rag_context, turn.citations = _format_rag_context_block(
        turn.retrieved_chunks, turn.rag_ceiling
    )
    yield _observation_step(turn)
    if turn.citations:
        yield _sse("citations", {"citations": turn.citations})

    synthesis_messages = to_langchain_messages(turn.context_messages) + [
        SystemMessage(
            content=(
                "## RETRIEVED KNOWLEDGE BASE CONTEXT (LanceDB Hybrid Search)\n"
                f"{rag_context}\n\n"
                "Use the retrieved context above to answer the user's question accurately. "
                "Cite sources in brackets like [filename.ext] or [URL]. If the retrieved context "
                "does not contain the answer, state that clearly."
            )
        ),
        HumanMessage(content=turn.user_message),
    ]

    full_answer = ""
    try:
        async for chunk in llm.astream(synthesis_messages):
            if await turn.request.is_disconnected():
                break
            delta = str(chunk.content or "")
            if delta:
                full_answer += delta
                yield _sse("token", {"delta": delta})
    except Exception as stream_error:
        if full_answer:
            suffix = (
                f"\n\n⚠️ **Stream interrupted ({turn.provider}):** `{str(stream_error)}`"
            )
            full_answer += suffix
            yield _sse("token", {"delta": suffix})
        else:
            full_answer = _synthesize_local_fallback_response(
                user_message=turn.user_message,
                citations=turn.citations,
                chunks=turn.retrieved_chunks,
                did_retrieve=turn.did_retrieve,
                provider=turn.provider,
                model=turn.model_name,
                upstream_error=str(stream_error),
            )
            yield _sse("token", {"delta": full_answer})

    async for event in _finish_turn(
        turn, full_answer or "[Generation stopped]", compact=True
    ):
        yield event


async def _stream_live_turn(
    turn: ChatTurn,
    llm: ChatOpenAI,
    llm_with_tools: Any,
    lc_messages: List[Any],
) -> AsyncGenerator[str, None]:
    tool_invocations: List[Dict[str, Any]] = []
    fatal_upstream_error: Optional[str] = None
    direct_ai_message: Optional[AIMessage] = None

    try:
        decision = await llm_with_tools.ainvoke(lc_messages)
        if getattr(decision, "tool_calls", None):
            for tool_call in decision.tool_calls:
                arguments = tool_call.get("args") or {}
                tool_invocations.append(
                    {
                        "reasoning": arguments.get("reasoning")
                        or str(decision.content or "").strip()
                        or "Query requires knowledge base lookup.",
                        "query": arguments.get("query") or turn.user_message,
                        "target_source": arguments.get("target_source"),
                    }
                )
        elif turn.target_docs:
            tool_invocations.append(
                {
                    "reasoning": (
                        f"User explicitly targeted {', '.join(turn.target_docs)}; "
                        "querying LanceDB with document filter."
                    ),
                    "query": turn.user_message,
                    "target_source": None,
                }
            )
        else:
            direct_ai_message = decision
    except Exception as llm_error:
        if _is_fatal_upstream_error(llm_error):
            fatal_upstream_error = str(llm_error)
        else:
            # Endpoint/model does not support tools; fall back to heuristic routing.
            should_retrieve, reason = _should_auto_retrieve_heuristic(
                turn.user_message, turn.target_docs
            )
            if should_retrieve:
                tool_invocations.append(
                    {
                        "reasoning": f"{reason} (Model tool-call fallback)",
                        "query": turn.user_message,
                        "target_source": None,
                    }
                )

    if fatal_upstream_error is not None:
        async for event in _stream_fatal_fallback(turn, fatal_upstream_error):
            yield event
        return

    if not tool_invocations:
        async for event in _stream_direct_answer(
            turn, llm, lc_messages, direct_ai_message
        ):
            yield event
        return

    async for event in _stream_retrieval_answer(turn, llm, tool_invocations):
        yield event


async def _stream_local_fallback(turn: ChatTurn) -> AsyncGenerator[str, None]:
    should_retrieve, reason = _should_auto_retrieve_heuristic(
        turn.user_message, turn.target_docs
    )
    yield _record_step(turn, "THOUGHT", detail=reason)
    await asyncio.sleep(0.05)

    if should_retrieve:
        turn.did_retrieve = True
        filter_description = f" | @filter={turn.target_docs}" if turn.target_docs else ""
        yield _record_step(
            turn,
            "ACTION",
            tool="search_knowledge_base",
            query=turn.user_message,
            target_docs=turn.target_docs,
            detail=f'search_knowledge_base(query="{turn.user_message}"{filter_description})',
        )
        await asyncio.sleep(0.05)

        turn.retrieved_chunks, retrieval_error = await _search(turn)
        if retrieval_error:
            yield _record_step(
                turn,
                "ERROR",
                detail=f"Knowledge base retrieval failed: {retrieval_error}",
            )
        _, turn.citations = _format_rag_context_block(
            turn.retrieved_chunks, turn.rag_ceiling
        )
        yield _observation_step(turn)
        if turn.citations:
            yield _sse("citations", {"citations": turn.citations})

    synthesized = _synthesize_local_fallback_response(
        user_message=turn.user_message,
        citations=turn.citations,
        chunks=turn.retrieved_chunks,
        did_retrieve=turn.did_retrieve,
        provider=turn.provider,
        model=turn.model_name,
    )

    accumulated = ""
    step_size = 14
    for index in range(0, len(synthesized), step_size):
        if await turn.request.is_disconnected():
            break
        piece = synthesized[index : index + step_size]
        accumulated += piece
        yield _sse("token", {"delta": piece})
        await asyncio.sleep(0.01)

    async for event in _finish_turn(
        turn, accumulated or "[Generation stopped]", compact=True
    ):
        yield event


async def run_react_chat_stream(
    request: Request,
    session_id: str,
    user_message: str,
    target_docs: List[str],
    use_global_docs: bool = True,
    use_memory: bool = True,
    linked_session_ids: Optional[List[str]] = None,
    header_api_key: Optional[str] = None,
    user_id: str = "default",
    idempotency_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Executes the ReAct loop and streams thoughts, tool calls, observations,
    citations, and final response tokens via Server-Sent Events.
    """
    try:
        settings = store.get_settings(user_id=user_id, include_decrypted_key=True)
        api_key = (header_api_key or "").strip() or settings.get("_decrypted_api_key", "")

        session = store.get_session(session_id)
        if not session:
            session = store.create_session(
                title=user_message[:48] or "New Chat", user_id=user_id
            )
            session_id = session["id"]

        if len(session.get("messages", [])) == 0 and session.get("title") in (
            "New Session",
            "New Chat",
        ):
            clean_title = re.sub(r"\s+", " ", user_message).strip()[:42]
            if clean_title:
                store.update_session(session_id, title=clean_title)
                yield _sse(
                    "session_titled",
                    {"session_id": session_id, "title": clean_title},
                )

        if use_memory:
            for fact in _extract_candidate_memory_facts(user_message):
                memory = store.add_memory(
                    fact=fact,
                    category="extracted",
                    source_session_id=session_id,
                    user_id=user_id,
                )
                yield _sse("memory_extracted", {"memory": memory})

        context_messages, budget_info = assemble_tiered_context(
            session=session,
            settings=settings,
            use_memory=use_memory,
            linked_session_ids=linked_session_ids,
            user_id=user_id,
        )

        user_message_record = store.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        yield _sse(
            "user_message_saved",
            {"message": user_message_record, "budget": budget_info},
        )

        turn = ChatTurn(
            request=request,
            session_id=session_id,
            user_message=user_message,
            target_docs=list(target_docs),
            use_global_docs=use_global_docs,
            use_memory=use_memory,
            linked_session_ids=linked_session_ids,
            user_id=user_id,
            settings=settings,
            api_key=api_key,
            provider=settings.get("provider", "openrouter"),
            base_url=settings.get("base_url", "https://openrouter.ai/api/v1"),
            model_name=settings.get("model", "openai/gpt-4o-mini"),
            temperature=float(settings.get("temperature", 0.2)),
            max_tokens=int(settings.get("max_tokens", 2048)),
            top_k=int(settings.get("top_k", 5)),
            similarity_threshold=float(settings.get("similarity_threshold", 0.15)),
            dedup_threshold=float(settings.get("dedup_threshold", 0.92)),
            rag_ceiling=int(settings.get("rag_token_ceiling", 1500)),
            embedding_model=settings.get("embedding_model", "all-MiniLM-L6-v2"),
            context_messages=context_messages,
        )

        if turn.has_live_llm:
            llm = _build_llm(turn)
            llm_with_tools = llm.bind_tools(RETRIEVAL_LC_TOOLS)
            lc_messages = to_langchain_messages(context_messages) + [
                HumanMessage(content=_build_user_prompt(turn))
            ]
            async for event in _stream_live_turn(
                turn, llm, llm_with_tools, lc_messages
            ):
                yield event
        else:
            async for event in _stream_local_fallback(turn):
                yield event
    finally:
        if idempotency_key:
            store.release_idempotency(idempotency_key)
