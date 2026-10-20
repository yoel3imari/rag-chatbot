"""Heuristics for tool routing, error classification, and memory extraction."""

import re
from typing import List, Tuple


def _is_fatal_upstream_error(exc: Exception) -> bool:
    """Returns True for authentication, permission, not-found, rate-limit, or connection failures."""
    class_name = type(exc).__name__.lower()
    message = str(exc).lower()
    if any(
        keyword in class_name
        for keyword in (
            "authentication",
            "permission",
            "notfound",
            "connection",
            "ratelimit",
            "timeout",
        )
    ):
        return True
    if any(
        code in message
        for code in (
            "401",
            "403",
            "404",
            "429",
            "invalid api key",
            "incorrect api key",
            "unauthorized",
        )
    ):
        return True
    return False


def _should_auto_retrieve_heuristic(
    user_message: str, target_docs: List[str]
) -> Tuple[bool, str]:
    """Heuristic ReAct classifier used for fallback or when @document is explicitly targeted."""
    if target_docs:
        return True, f"User explicitly targeted {', '.join(target_docs)} via @mention filter."
    cleaned_message = user_message.strip().lower()
    chitchat_patterns = [
        r"^((hi|hello|hey|hi there|hello there|good morning|good afternoon|good evening|yo|sup|howdy)[\s!.,?]*)+$",
        r"^((thanks|thank you|thx|ok|okay|got it|cool|awesome|great)[\s!.,?]*)+$",
        r"^(who are you|what are you|how are you|what can you do)[\s!.,?]*$",
    ]
    for pattern in chitchat_patterns:
        if re.match(pattern, cleaned_message):
            return (
                False,
                "Direct conversational greeting/chitchat; no document retrieval required.",
            )
    if re.match(r"^[\d\s\+\-\*\/\(\)\.\^\=whatis]+[\?]?$", cleaned_message) and any(
        character.isdigit() for character in cleaned_message
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
    for pattern in patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)
        if match:
            extracted = match.group(0).strip().rstrip(".")
            if len(extracted) >= 8:
                facts.append(extracted)
    return facts
