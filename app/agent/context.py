"""Tiered prompt assembly and rolling-summary compaction."""

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import KEYLESS_PROVIDERS
from app.store import estimate_tokens, store


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    approximate_chars = max(100, int(max_tokens * 3.6))
    return text[:approximate_chars] + "..."


def _build_memory_block(memories: List[Dict[str, Any]]) -> Optional[str]:
    if not memories:
        return None
    lines = [f"- {memory['fact']}" for memory in memories[:25]]
    block = "## GLOBAL MEMORY LEDGER (Persistent User Facts & Constraints)\n" + "\n".join(
        lines
    )
    return _truncate_to_token_budget(block, 400)


def _build_linked_session_block(
    session: Dict[str, Any], linked_session_ids: Optional[List[str]]
) -> Optional[str]:
    all_linked_ids = list(
        dict.fromkeys((session.get("linked_sessions") or []) + (linked_session_ids or []))
    )
    if not all_linked_ids:
        return None

    linked_parts = []
    for linked_id in all_linked_ids:
        if linked_id == session["id"]:
            continue
        linked_session = store.get_session(linked_id)
        if not linked_session:
            continue
        summary = linked_session.get("summary") or ""
        if not summary and linked_session.get("messages"):
            snippets = [
                f"{message['role'].upper()}: {message['content'][:160]}"
                for message in linked_session["messages"][-4:]
            ]
            summary = " | ".join(snippets)
        if summary:
            linked_parts.append(f"- Session [{linked_session['title']}]: {summary}")

    if not linked_parts:
        return None
    block = "## LINKED SESSION CONTEXT\n" + "\n".join(linked_parts)
    return _truncate_to_token_budget(block, 400)


def assemble_tiered_context(
    session: Dict[str, Any],
    settings: Dict[str, Any],
    use_memory: bool = True,
    linked_session_ids: Optional[List[str]] = None,
    user_id: str = "default",
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Constructs prompt payload by tiered budget allocation and prompt-caching prefix order:
    System Prompt -> Global Memory -> Linked Session Context -> Rolling Summary -> Recent Turns.
    """
    max_context = int(settings.get("max_context_tokens", 8192))
    reserved_output = int(settings.get("reserved_output_tokens", 2048))
    rag_ceiling = int(settings.get("rag_token_ceiling", 1500))
    compaction_mode = settings.get("compaction_mode", "Auto-Summarize")
    window_turns = int(settings.get("sliding_window_turns", 10))

    available_input_budget = max(1024, max_context - reserved_output)

    # 1. Static system prefix (prompt-caching aligned).
    system_prompt = settings.get("system_prompt", "").strip()
    system_blocks = [system_prompt]

    # 2. Cross-session global memory ledger.
    memory_tokens = 0
    if use_memory:
        memory_block = _build_memory_block(store.list_memories(user_id=user_id))
        if memory_block:
            memory_tokens = estimate_tokens(memory_block)
            system_blocks.append(memory_block)

    # 3. Linked session summaries (@session / /link-session).
    linked_tokens = 0
    linked_block = _build_linked_session_block(session, linked_session_ids)
    if linked_block:
        linked_tokens = estimate_tokens(linked_block)
        system_blocks.append(linked_block)

    # 4. Rolling intra-session summary.
    summary_text = (session.get("summary") or "").strip()
    summary_tokens = 0
    if summary_text:
        summary_block = (
            f"## ROLLING SESSION SUMMARY (Compressed Prior Context)\n{summary_text}"
        )
        summary_block = _truncate_to_token_budget(summary_block, 600)
        summary_tokens = estimate_tokens(summary_block)
        system_blocks.append(summary_block)

    combined_system = "\n\n".join(system_blocks)
    system_tokens = estimate_tokens(combined_system)

    # 5. Recent conversation turns, excluding pruned messages.
    active_messages = [
        message
        for message in session.get("messages", [])
        if message.get("included_in_context", True)
    ]
    if compaction_mode == "Sliding Window":
        active_messages = active_messages[-(window_turns * 2) :]

    history_budget = max(
        512,
        available_input_budget
        - system_tokens
        - min(rag_ceiling, available_input_budget // 3),
    )

    selected_turns: List[Dict[str, str]] = []
    used_history_tokens = 0
    for message in reversed(active_messages):
        turn_cost = int(message.get("tokens") or estimate_tokens(message["content"]))
        if used_history_tokens + turn_cost > history_budget and selected_turns:
            break
        selected_turns.append(
            {"role": message["role"], "content": message["content"]}
        )
        used_history_tokens += turn_cost
    selected_turns.reverse()

    assembled = [{"role": "system", "content": combined_system}] + selected_turns
    budget_breakdown = {
        "system_tokens": system_tokens,
        "memory_tokens": memory_tokens,
        "linked_tokens": linked_tokens,
        "summary_tokens": summary_tokens,
        "history_tokens": used_history_tokens,
        "available_input_budget": available_input_budget,
        "max_context_tokens": max_context,
    }
    return assembled, budget_breakdown


async def maybe_compact_session_summary(
    session_id: str,
    settings: Dict[str, Any],
    api_key: str,
) -> Optional[str]:
    """
    Compresses older messages into the rolling session summary once token usage reaches
    75% of the configured context limit (or the sliding window overflows).
    """
    session = store.get_session(session_id)
    if not session:
        return None
    compaction_mode = settings.get("compaction_mode", "Auto-Summarize")
    if compaction_mode != "Auto-Summarize":
        return None

    max_context = int(settings.get("max_context_tokens", 8192))
    reserved_output = int(settings.get("reserved_output_tokens", 2048))
    effective_limit = max(1024, max_context - reserved_output)
    threshold_tokens = int(effective_limit * 0.75)

    active_messages = [
        message
        for message in session.get("messages", [])
        if message.get("included_in_context", True)
    ]
    total_tokens = sum(
        int(message.get("tokens") or estimate_tokens(message["content"]))
        for message in active_messages
    )
    window_turns = int(settings.get("sliding_window_turns", 10))

    if total_tokens < threshold_tokens and len(active_messages) <= window_turns * 2:
        return None

    cutoff = max(2, len(active_messages) - window_turns)
    older_messages = active_messages[:cutoff]
    if not older_messages:
        return None

    existing_summary = session.get("summary") or ""
    transcript = "\n".join(
        f"{message['role'].upper()}: {message['content'][:400]}"
        for message in older_messages
    )

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
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Compress the conversation history into a concise, high-density "
                            "factual summary (max 150 words). Preserve key decisions, entities, "
                            "and user constraints."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Prior Summary:\n{existing_summary}\n\n"
                            f"New Turns to Compress:\n{transcript}"
                        )
                    ),
                ]
            )
            new_summary = str(response.content or "").strip()
        except Exception:
            new_summary = ""

    if not new_summary:
        bullets = []
        if existing_summary:
            bullets.append(existing_summary)
        for message in older_messages[-6:]:
            snippet = re.sub(r"\s+", " ", message["content"]).strip()[:140]
            bullets.append(f"[{message['role'].upper()}] {snippet}")
        new_summary = " • ".join(bullets)[:1000]

    for message in older_messages:
        store.toggle_message_context(message["id"], False)

    store.update_session(session_id, summary=new_summary)
    return new_summary
