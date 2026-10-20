"""Offline/local answer synthesis when no live LLM is available."""

from typing import Any, Dict, List, Optional


def _synthesize_local_fallback_response(
    user_message: str,
    citations: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    did_retrieve: bool,
    provider: str,
    model: str,
    upstream_error: Optional[str] = None,
) -> str:
    """Provides a clean, accurate response in Local Fallback Mode or after an upstream error."""
    header_notice = ""
    if upstream_error:
        header_notice = (
            f"⚠️ **Upstream LLM Error ({provider} / `{model}`):** `{upstream_error}`\n\n"
        )

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
    for index, chunk in enumerate(chunks[:4]):
        lines.append(
            f"**{index + 1}. [{chunk['source_name']}]**\n"
            f"> {chunk['text'].strip()}\n"
        )
    if not upstream_error:
        lines.append(
            f"\n*(Add an API key for `{provider}` in **Settings → LLM** for full AI-generated answers.)*"
        )
    return "\n".join(lines)
