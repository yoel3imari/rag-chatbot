import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from fastapi import Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import KEYLESS_PROVIDERS
from app.embeddings import embedding_engine
from app.store import estimate_tokens, store
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


def to_langchain_messages(raw_messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts dict messages into LangChain BaseMessage instances."""
    lc_msgs: List[BaseMessage] = []
    for m in raw_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_msgs.append(AIMessage(content=content))
        else:
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs


def _is_fatal_upstream_error(exc: Exception) -> bool:
    """Returns True if the exception is an authentication, permission, not-found, or connection failure."""
    cls_name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(
        k in cls_name
        for k in ("authentication", "permission", "notfound", "connection", "ratelimit", "timeout")
    ):
        return True
    if any(code in msg for code in ("401", "403", "404", "429", "invalid api key", "incorrect api key", "unauthorized")):
        return True
    return False


async def _safe_hybrid_search(**search_kwargs: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
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


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    approx_chars = max(100, int(max_tokens * 3.6))
    return text[:approx_chars] + "..."


def _should_auto_retrieve_heuristic(user_message: str, target_docs: List[str]) -> Tuple[bool, str]:
    """Heuristic ReAct classifier used for fallback or when @document is explicitly targeted."""
    if target_docs:
        return True, f"User explicitly targeted {', '.join(target_docs)} via @mention filter."
    msg_clean = user_message.strip().lower()
    chitchat_patterns = [
        r"^((hi|hello|hey|hi there|hello there|good morning|good afternoon|good evening|yo|sup|howdy)[\s!.,?]*)+$",
        r"^((thanks|thank you|thx|ok|okay|got it|cool|awesome|great)[\s!.,?]*)+$",
        r"^(who are you|what are you|how are you|what can you do)[\s!.,?]*$",
    ]
    for pat in chitchat_patterns:
        if re.match(pat, msg_clean):
            return False, "Direct conversational greeting/chitchat; no document retrieval required."
    if re.match(r"^[\d\s\+\-\*\/\(\)\.\^\=whatis]+[\?]?$", msg_clean) and any(
        c.isdigit() for c in msg_clean
    ):
        return False, "Pure mathematical calculation; answering directly via general logic."
    return True, "Query may reference indexed knowledge base documents or domain context."


def _extract_candidate_memory_facts(user_message: str) -> List[str]:
    """Extracts explicit durable user preferences/facts for the Cross-Session Memory Ledger."""
    facts: List[str] = []
    patterns = [
        r"(?:remember that|please remember|note that|for future reference[:,]?)\s+(.{10,200})",
        r"(?:my name is|i work as|my role is|my team is|our company is)\s+([^.!?\n]{3,120})",
        r"(?:always respond in|always format|prefer responses to be)\s+([^.!?\n]{5,120})",
    ]
    for pat in patterns:
        m = re.search(pat, user_message, flags=re.IGNORECASE)
        if m:
            extracted = m.group(0).strip().rstrip(".")
            if len(extracted) >= 8:
                facts.append(extracted)
    return facts


def assemble_tiered_context(
    session: Dict[str, Any],
    settings: Dict[str, Any],
    use_memory: bool = True,
    linked_session_ids: Optional[List[str]] = None,
    user_id: str = "default",
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Constructs prompt payload strictly by tiered budget allocation and prompt-caching prefix order:
    System Prompt -> Global Memory -> Linked Session Context -> Rolling Summary -> Recent Turns.
    """
    max_ctx = int(settings.get("max_context_tokens", 8192))
    reserved_out = int(settings.get("reserved_output_tokens", 2048))
    rag_ceiling = int(settings.get("rag_token_ceiling", 1500))
    compaction_mode = settings.get("compaction_mode", "Auto-Summarize")
    window_turns = int(settings.get("sliding_window_turns", 10))

    available_input_budget = max(1024, max_ctx - reserved_out)

    # 1. Static System Prefix (Prompt Caching Aligned)
    sys_prompt = settings.get("system_prompt", "").strip()
    sys_blocks = [sys_prompt]

    # 2. Cross-Session Global Memory Ledger
    mem_tokens = 0
    if use_memory:
        memories = store.list_memories(user_id=user_id)
        if memories:
            mem_lines = [f"- {m['fact']}" for m in memories[:25]]
            mem_block = "## GLOBAL MEMORY LEDGER (Persistent User Facts & Constraints)\n" + "\n".join(mem_lines)
            mem_block = _truncate_to_token_budget(mem_block, 400)
            mem_tokens = estimate_tokens(mem_block)
            sys_blocks.append(mem_block)

    # 3. Linked Session Summaries (@session / /link-session)
    all_linked_ids = list(
        dict.fromkeys((session.get("linked_sessions") or []) + (linked_session_ids or []))
    )
    linked_tokens = 0
    if all_linked_ids:
        linked_parts = []
        for lid in all_linked_ids:
            if lid == session["id"]:
                continue
            lsess = store.get_session(lid)
            if lsess:
                l_sum = lsess.get("summary") or ""
                if not l_sum and lsess.get("messages"):
                    snippets = [
                        f"{m['role'].upper()}: {m['content'][:160]}"
                        for m in lsess["messages"][-4:]
                    ]
                    l_sum = " | ".join(snippets)
                if l_sum:
                    linked_parts.append(f"- Session [{lsess['title']}]: {l_sum}")
        if linked_parts:
            linked_block = "## LINKED SESSION CONTEXT\n" + "\n".join(linked_parts)
            linked_block = _truncate_to_token_budget(linked_block, 400)
            linked_tokens = estimate_tokens(linked_block)
            sys_blocks.append(linked_block)

    # 4. Rolling Intra-Session Summary
    summary_text = (session.get("summary") or "").strip()
    summary_tokens = 0
    if summary_text:
        sum_block = f"## ROLLING SESSION SUMMARY (Compressed Prior Context)\n{summary_text}"
        sum_block = _truncate_to_token_budget(sum_block, 600)
        summary_tokens = estimate_tokens(sum_block)
        sys_blocks.append(sum_block)

    combined_system = "\n\n".join(sys_blocks)
    sys_total_tokens = estimate_tokens(combined_system)

    # 5. Recent Conversation Turns (filtering out pruned messages where included_in_context == False)
    active_messages = [
        m for m in session.get("messages", []) if m.get("included_in_context", True)
    ]

    if compaction_mode == "Sliding Window":
        active_messages = active_messages[-(window_turns * 2) :]

    history_budget = max(
        512, available_input_budget - sys_total_tokens - min(rag_ceiling, available_input_budget // 3)
    )

    selected_turns: List[Dict[str, str]] = []
    used_history_tokens = 0
    for m in reversed(active_messages):
        t_cost = int(m.get("tokens") or estimate_tokens(m["content"]))
        if used_history_tokens + t_cost > history_budget and selected_turns:
            break
        selected_turns.append({"role": m["role"], "content": m["content"]})
        used_history_tokens += t_cost

    selected_turns.reverse()

    assembled = [{"role": "system", "content": combined_system}] + selected_turns
    budget_breakdown = {
        "system_tokens": sys_total_tokens,
        "memory_tokens": mem_tokens,
        "linked_tokens": linked_tokens,
        "summary_tokens": summary_tokens,
        "history_tokens": used_history_tokens,
        "available_input_budget": available_input_budget,
        "max_context_tokens": max_ctx,
    }
    return assembled, budget_breakdown


async def maybe_compact_session_summary(
    session_id: str,
    settings: Dict[str, Any],
    api_key: str,
) -> Optional[str]:
    """
    Checks if session token count >= 75% of configured context limit (or exceeds sliding window in Auto-Summarize mode)
    and compresses older messages into the rolling Session Summary block via LangChain ChatOpenAI.
    """
    session = store.get_session(session_id)
    if not session:
        return None
    compaction_mode = settings.get("compaction_mode", "Auto-Summarize")
    if compaction_mode != "Auto-Summarize":
        return None

    max_ctx = int(settings.get("max_context_tokens", 8192))
    reserved_out = int(settings.get("reserved_output_tokens", 2048))
    effective_limit = max(1024, max_ctx - reserved_out)
    threshold_tokens = int(effective_limit * 0.75)

    active_msgs = [m for m in session.get("messages", []) if m.get("included_in_context", True)]
    total_tokens = sum(int(m.get("tokens") or estimate_tokens(m["content"])) for m in active_msgs)
    window_turns = int(settings.get("sliding_window_turns", 10))

    if total_tokens < threshold_tokens and len(active_msgs) <= window_turns * 2:
        return None

    cutoff = max(2, len(active_msgs) - window_turns)
    older_msgs = active_msgs[:cutoff]
    if not older_msgs:
        return None

    existing_summary = session.get("summary") or ""
    transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in older_msgs)

    new_summary = ""
    provider = settings.get("provider", "openrouter")
    if api_key or provider in KEYLESS_PROVIDERS:
        try:
            llm = ChatOpenAI(
                api_key=api_key or "not-needed",
                base_url=settings.get("base_url", "https://openrouter.ai/api/v1"),
                model=settings.get("model", "openai/gpt-4o-mini"),
                temperature=0.1,
                max_tokens=250,
                timeout=15.0,
                max_retries=1,
            )
            resp = await llm.ainvoke(
                [
                    SystemMessage(
                        content="Compress the conversation history into a concise, high-density factual summary (max 150 words). Preserve key decisions, entities, and user constraints."
                    ),
                    HumanMessage(
                        content=f"Prior Summary:\n{existing_summary}\n\nNew Turns to Compress:\n{transcript}"
                    ),
                ]
            )
            new_summary = str(resp.content or "").strip()
        except Exception:
            new_summary = ""

    if not new_summary:
        bullets = []
        if existing_summary:
            bullets.append(existing_summary)
        for m in older_msgs[-6:]:
            snippet = re.sub(r"\s+", " ", m["content"]).strip()[:140]
            bullets.append(f"[{m['role'].upper()}] {snippet}")
        new_summary = " • ".join(bullets)[:1000]

    for m in older_msgs:
        store.toggle_message_context(m["id"], False)

    store.update_session(session_id, summary=new_summary)
    return new_summary


def _format_rag_context_block(
    chunks: List[Dict[str, Any]], rag_token_ceiling: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """Formats retrieved chunks within the hard RAG token ceiling."""
    if not chunks:
        return "No matching documents found in LanceDB knowledge base.", []

    formatted_parts: List[str] = []
    kept_citations: List[Dict[str, Any]] = []
    used_tokens = 0

    for idx, c in enumerate(chunks):
        block = (
            f"[Source {idx + 1}: {c['source_name']} | Chunk #{c['chunk_index']} | "
            f"Score: {c['hybrid_score']} ({c['match_type']})]\n{c['text']}"
        )
        b_tokens = estimate_tokens(block)
        if used_tokens + b_tokens > rag_token_ceiling and formatted_parts:
            break
        formatted_parts.append(block)
        used_tokens += b_tokens
        kept_citations.append(
            {
                "index": idx + 1,
                "doc_id": c["doc_id"],
                "source_name": c["source_name"],
                "source_type": c["source_type"],
                "scope": c["scope"],
                "chunk_index": c["chunk_index"],
                "similarity": c["similarity"],
                "hybrid_score": c["hybrid_score"],
                "match_type": c["match_type"],
                "preview": c["text"][:240] + ("..." if len(c["text"]) > 240 else ""),
            }
        )

    return "\n\n---\n\n".join(formatted_parts), kept_citations


def _synthesize_local_fallback_response(
    user_message: str,
    citations: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    did_retrieve: bool,
    provider: str,
    model: str,
    upstream_error: Optional[str] = None,
) -> str:
    """Provides a clean, accurate response when in Local Fallback Mode or when an upstream error occurred."""
    header_notice = ""
    if upstream_error:
        header_notice = f"⚠️ **Upstream LLM Error ({provider} / `{model}`):** `{upstream_error}`\n\n"

    if not did_retrieve:
        if upstream_error:
            return (
                f"{header_notice}"
                "Please verify your API Key, Base URL, and Model in **Settings → LLM**."
            )
        return (
            f"Hello! No API key is currently configured for `{provider}` (`{model}`).\n\n"
            "You can upload PDF, TXT, or Markdown files—or add URLs—directly from the chat bar or **Settings**, "
            "filter documents with `@`, and add your API key in **Settings → LLM**."
        )

    if not chunks:
        return (
            f"{header_notice}"
            "I searched your knowledge base, but no matching documents were found for your query.\n\n"
            "> **Tip:** Upload a `.PDF`, `.TXT`, or `.MD` document or add a URL using the buttons above the prompt bar."
        )

    lines = [
        f"{header_notice}Here are the most relevant excerpts from your documents for **{user_message}**:\n"
    ]
    for i, c in enumerate(chunks[:4]):
        lines.append(
            f"**{i + 1}. [{c['source_name']}]**\n"
            f"> {c['text'].strip()}\n"
        )
    if not upstream_error:
        lines.append(
            f"\n*(Add an API key for `{provider}` in **Settings → LLM** for full AI-generated answers.)*"
        )
    return "\n".join(lines)


def _sse(event_type: str, payload: Dict[str, Any]) -> str:
    data = {"type": event_type, **payload}
    return f"data: {json.dumps(data)}\n\n"


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
    Executes the LangChain ReAct (Reasoning + Acting) loop and streams thoughts, tool calls,
    observations, citations, and final response tokens via Server-Sent Events (SSE).
    """
    try:
        settings = store.get_settings(user_id=user_id, include_decrypted_key=True)
        api_key = (header_api_key or "").strip() or settings.get("_decrypted_api_key", "")
        provider = settings.get("provider", "openrouter")
        base_url = settings.get("base_url", "https://openrouter.ai/api/v1")
        model_name = settings.get("model", "openai/gpt-4o-mini")
        temperature = float(settings.get("temperature", 0.2))
        max_tokens = int(settings.get("max_tokens", 2048))
        top_k = int(settings.get("top_k", 5))
        sim_threshold = float(settings.get("similarity_threshold", 0.15))
        dedup_threshold = float(settings.get("dedup_threshold", 0.92))
        rag_ceiling = int(settings.get("rag_token_ceiling", 1500))
        emb_model = settings.get("embedding_model", "all-MiniLM-L6-v2")

        session = store.get_session(session_id)
        if not session:
            session = store.create_session(title=user_message[:48] or "New Chat", user_id=user_id)
            session_id = session["id"]

        if len(session.get("messages", [])) == 0 and session.get("title") in ("New Session", "New Chat"):
            clean_title = re.sub(r"\s+", " ", user_message).strip()[:42]
            if clean_title:
                store.update_session(session_id, title=clean_title)
                yield _sse("session_titled", {"session_id": session_id, "title": clean_title})

        if use_memory:
            extracted_facts = _extract_candidate_memory_facts(user_message)
            for fact in extracted_facts:
                mem_obj = store.add_memory(
                    fact=fact, category="extracted", source_session_id=session_id, user_id=user_id
                )
                yield _sse("memory_extracted", {"memory": mem_obj})

        context_messages, budget_info = assemble_tiered_context(
            session=session,
            settings=settings,
            use_memory=use_memory,
            linked_session_ids=linked_session_ids,
            user_id=user_id,
        )

        user_msg_obj = store.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        yield _sse("user_message_saved", {"message": user_msg_obj, "budget": budget_info})

        react_steps: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        retrieved_chunks: List[Dict[str, Any]] = []
        did_retrieve = False

        has_live_llm = bool(api_key) or (provider in KEYLESS_PROVIDERS)

        if has_live_llm:
            llm = ChatOpenAI(
                api_key=api_key or "not-needed",
                base_url=base_url,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=45.0,
                max_retries=2,
            )
            llm_with_tools = llm.bind_tools(RETRIEVAL_LC_TOOLS)

            user_prompt_for_llm = user_message
            if target_docs:
                user_prompt_for_llm = (
                    f"[Targeted Document Filter Active: {', '.join(target_docs)}]\n"
                    f"User Query: {user_message}\n"
                    "(Note: Please call `search_knowledge_base` to inspect the targeted document(s).)"
                )

            lc_messages = to_langchain_messages(context_messages) + [
                HumanMessage(content=user_prompt_for_llm)
            ]

            tool_invocations: List[Dict[str, Any]] = []
            fatal_upstream_err: Optional[str] = None
            direct_ai_msg: Optional[AIMessage] = None

            try:
                decision_msg = await llm_with_tools.ainvoke(lc_messages)
                if getattr(decision_msg, "tool_calls", None):
                    for tc in decision_msg.tool_calls:
                        args = tc.get("args") or {}
                        tool_invocations.append(
                            {
                                "reasoning": args.get("reasoning")
                                or str(decision_msg.content or "").strip()
                                or "Query requires knowledge base lookup.",
                                "query": args.get("query") or user_message,
                                "target_source": args.get("target_source"),
                            }
                        )
                elif target_docs:
                    tool_invocations.append(
                        {
                            "reasoning": f"User explicitly targeted {', '.join(target_docs)}; querying LanceDB with document filter.",
                            "query": user_message,
                            "target_source": None,
                        }
                    )
                else:
                    direct_ai_msg = decision_msg
            except Exception as llm_err:
                if _is_fatal_upstream_error(llm_err):
                    fatal_upstream_err = str(llm_err)
                else:
                    # Endpoint/model does not support `tools`; fall back to heuristic ReAct decision
                    should_ret, reason = _should_auto_retrieve_heuristic(user_message, target_docs)
                    if should_ret:
                        tool_invocations.append(
                            {
                                "reasoning": f"{reason} (Model tool-call fallback)",
                                "query": user_message,
                                "target_source": None,
                            }
                        )

            # Handle fatal upstream auth/connection error transparently
            if fatal_upstream_err is not None:
                err_step = {
                    "step": "ERROR",
                    "detail": f"Upstream provider error ({provider}): {fatal_upstream_err[:160]}",
                }
                react_steps.append(err_step)
                yield _sse("react_step", err_step)

                should_ret, _ = _should_auto_retrieve_heuristic(user_message, target_docs)
                if should_ret:
                    did_retrieve = True
                    retrieved_chunks, ret_err = await _safe_hybrid_search(
                        query=user_message,
                        user_id=user_id,
                        session_id=session_id,
                        use_global_docs=use_global_docs,
                        target_docs=target_docs if target_docs else None,
                        top_k=top_k,
                        similarity_threshold=sim_threshold,
                        dedup_threshold=dedup_threshold,
                        embedding_model=emb_model,
                    )
                    if ret_err:
                        ret_err_step = {
                            "step": "ERROR",
                            "detail": f"Knowledge base retrieval failed: {ret_err}",
                        }
                        react_steps.append(ret_err_step)
                        yield _sse("react_step", ret_err_step)
                    _, citations = _format_rag_context_block(retrieved_chunks, rag_ceiling)
                    if citations:
                        yield _sse("citations", {"citations": citations})

                err_response = _synthesize_local_fallback_response(
                    user_message=user_message,
                    citations=citations,
                    chunks=retrieved_chunks,
                    did_retrieve=did_retrieve,
                    provider=provider,
                    model=model_name,
                    upstream_error=fatal_upstream_err,
                )
                yield _sse("token", {"delta": err_response})
                assistant_msg = store.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=err_response,
                    react_steps=react_steps,
                    citations=citations,
                )
                yield _sse("done", {"message": assistant_msg})
                return

            # Branch A: Direct answer without tool calls
            if not tool_invocations:
                thought_step = {
                    "step": "THOUGHT",
                    "detail": "Direct response selected (no knowledge base retrieval required).",
                }
                react_steps.append(thought_step)
                yield _sse("react_step", thought_step)

                direct_text = str(direct_ai_msg.content if direct_ai_msg else "").strip()
                if not direct_text:
                    # If model tool-call fallback took the non-retrieval path, stream direct completion from LLM
                    try:
                        async for chunk in llm.astream(lc_messages):
                            if await request.is_disconnected():
                                break
                            piece = str(chunk.content or "")
                            if piece:
                                direct_text += piece
                                yield _sse("token", {"delta": piece})
                    except Exception as stream_err:
                        err_suffix = f"\n\n⚠️ **Upstream LLM Error ({provider}):** `{str(stream_err)}`"
                        direct_text += err_suffix
                        yield _sse("token", {"delta": err_suffix})
                else:
                    chunk_step = max(6, len(direct_text) // 40)
                    for i in range(0, len(direct_text), chunk_step):
                        if await request.is_disconnected():
                            break
                        piece = direct_text[i : i + chunk_step]
                        yield _sse("token", {"delta": piece})
                        await asyncio.sleep(0.01)

                final_content = direct_text or "[Empty model response]"
                assistant_msg = store.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                    react_steps=react_steps,
                    citations=[],
                )
                yield _sse("done", {"message": assistant_msg})
                if not await request.is_disconnected():
                    new_sum = await maybe_compact_session_summary(session_id, settings, api_key)
                    if new_sum:
                        yield _sse("summary_updated", {"summary": new_sum})
                return

            # Branch B: Execute all tool calls (supporting parallel tool calls)
            did_retrieve = True
            combined_chunks: List[Dict[str, Any]] = []

            for inv in tool_invocations:
                t_reason = inv["reasoning"]
                t_query = inv["query"]
                t_target = inv["target_source"]

                effective_targets = list(target_docs)
                if t_target and t_target not in effective_targets:
                    effective_targets.append(t_target)

                step_thought = {"step": "THOUGHT", "detail": t_reason}
                react_steps.append(step_thought)
                yield _sse("react_step", step_thought)

                filter_desc = f" | @filter={effective_targets}" if effective_targets else ""
                step_action = {
                    "step": "ACTION",
                    "tool": "search_knowledge_base",
                    "query": t_query,
                    "target_docs": effective_targets,
                    "detail": f'search_knowledge_base(query="{t_query}"{filter_desc})',
                }
                react_steps.append(step_action)
                yield _sse("react_step", step_action)

                call_chunks, ret_err = await _safe_hybrid_search(
                    query=t_query,
                    user_id=user_id,
                    session_id=session_id,
                    use_global_docs=use_global_docs,
                    target_docs=effective_targets if effective_targets else None,
                    top_k=top_k,
                    similarity_threshold=sim_threshold,
                    dedup_threshold=dedup_threshold,
                    embedding_model=emb_model,
                )
                if ret_err:
                    ret_err_step = {
                        "step": "ERROR",
                        "detail": f"Knowledge base retrieval failed: {ret_err}",
                    }
                    react_steps.append(ret_err_step)
                    yield _sse("react_step", ret_err_step)
                else:
                    combined_chunks.extend(call_chunks)

            # Deduplicate across parallel tool calls by chunk ID and cosine similarity
            seen_ids = set()
            unique_chunks = []
            for c in combined_chunks:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    unique_chunks.append(c)
            retrieved_chunks = embedding_engine.deduplicate_chunks(
                unique_chunks, threshold=dedup_threshold
            )[:top_k]

            rag_context_str, citations = _format_rag_context_block(retrieved_chunks, rag_ceiling)
            step_obs = {
                "step": "OBSERVATION",
                "chunks_count": len(citations),
                "detail": (
                    f"Retrieved {len(citations)} hybrid chunks from LanceDB "
                    f"({', '.join(dict.fromkeys(c['source_name'] for c in citations)) or '0 matches'})."
                ),
            }
            react_steps.append(step_obs)
            yield _sse("react_step", step_obs)
            if citations:
                yield _sse("citations", {"citations": citations})

            # Synthesize grounded answer via LangChain streaming (dynamic RAG block placed after static prefix + history)
            synthesis_messages = to_langchain_messages(context_messages) + [
                SystemMessage(
                    content=(
                        f"## RETRIEVED KNOWLEDGE BASE CONTEXT (LanceDB Hybrid Search)\n{rag_context_str}\n\n"
                        "Use the retrieved context above to answer the user's question accurately. "
                        "Cite sources in brackets like [filename.ext] or [URL]. If the retrieved context "
                        "does not contain the answer, state that clearly."
                    )
                ),
                HumanMessage(content=user_message),
            ]

            full_answer = ""
            try:
                async for chunk in llm.astream(synthesis_messages):
                    if await request.is_disconnected():
                        break
                    delta = str(chunk.content or "")
                    if delta:
                        full_answer += delta
                        yield _sse("token", {"delta": delta})
            except Exception as stream_err:
                if full_answer:
                    # Preserve already streamed partial content!
                    err_suffix = f"\n\n⚠️ **Stream interrupted ({provider}):** `{str(stream_err)}`"
                    full_answer += err_suffix
                    yield _sse("token", {"delta": err_suffix})
                else:
                    fallback_txt = _synthesize_local_fallback_response(
                        user_message=user_message,
                        citations=citations,
                        chunks=retrieved_chunks,
                        did_retrieve=did_retrieve,
                        provider=provider,
                        model=model_name,
                        upstream_error=str(stream_err),
                    )
                    full_answer = fallback_txt
                    yield _sse("token", {"delta": full_answer})

            assistant_msg = store.add_message(
                session_id=session_id,
                role="assistant",
                content=full_answer or "[Generation stopped]",
                react_steps=react_steps,
                citations=citations,
            )
            yield _sse("done", {"message": assistant_msg})
            if not await request.is_disconnected():
                new_sum = await maybe_compact_session_summary(session_id, settings, api_key)
                if new_sum:
                    yield _sse("summary_updated", {"summary": new_sum})
            return

        # Local ReAct Fallback Mode (when no external LLM API key is configured yet)
        should_retrieve, reason = _should_auto_retrieve_heuristic(user_message, target_docs)
        step_thought = {"step": "THOUGHT", "detail": reason}
        react_steps.append(step_thought)
        yield _sse("react_step", step_thought)
        await asyncio.sleep(0.05)

        if should_retrieve:
            did_retrieve = True
            filter_desc = f" | @filter={target_docs}" if target_docs else ""
            step_action = {
                "step": "ACTION",
                "tool": "search_knowledge_base",
                "query": user_message,
                "target_docs": target_docs,
                "detail": f'search_knowledge_base(query="{user_message}"{filter_desc})',
            }
            react_steps.append(step_action)
            yield _sse("react_step", step_action)
            await asyncio.sleep(0.05)

            retrieved_chunks, ret_err = await _safe_hybrid_search(
                query=user_message,
                user_id=user_id,
                session_id=session_id,
                use_global_docs=use_global_docs,
                target_docs=target_docs if target_docs else None,
                top_k=top_k,
                similarity_threshold=sim_threshold,
                dedup_threshold=dedup_threshold,
                embedding_model=emb_model,
            )
            if ret_err:
                ret_err_step = {
                    "step": "ERROR",
                    "detail": f"Knowledge base retrieval failed: {ret_err}",
                }
                react_steps.append(ret_err_step)
                yield _sse("react_step", ret_err_step)
            _, citations = _format_rag_context_block(retrieved_chunks, rag_ceiling)
            step_obs = {
                "step": "OBSERVATION",
                "chunks_count": len(citations),
                "detail": (
                    f"Retrieved {len(citations)} hybrid chunks from LanceDB "
                    f"({', '.join(dict.fromkeys(c['source_name'] for c in citations)) or '0 matches'})."
                ),
            }
            react_steps.append(step_obs)
            yield _sse("react_step", step_obs)
            if citations:
                yield _sse("citations", {"citations": citations})

        synthesized = _synthesize_local_fallback_response(
            user_message=user_message,
            citations=citations,
            chunks=retrieved_chunks,
            did_retrieve=did_retrieve,
            provider=provider,
            model=model_name,
        )

        step_size = 14
        acc = ""
        for i in range(0, len(synthesized), step_size):
            if await request.is_disconnected():
                break
            piece = synthesized[i : i + step_size]
            acc += piece
            yield _sse("token", {"delta": piece})
            await asyncio.sleep(0.01)

        assistant_msg = store.add_message(
            session_id=session_id,
            role="assistant",
            content=acc or "[Generation stopped]",
            react_steps=react_steps,
            citations=citations,
        )
        yield _sse("done", {"message": assistant_msg})
        if not await request.is_disconnected():
            new_sum = await maybe_compact_session_summary(session_id, settings, api_key)
            if new_sum:
                yield _sse("summary_updated", {"summary": new_sum})
    finally:
        if idempotency_key:
            store.release_idempotency(idempotency_key)
