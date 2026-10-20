"""Heuristic token accounting shared by the backend and mirrored in the JS worker."""


def estimate_tokens(text: str) -> int:
    """Estimates tokens (~3.8 chars/token + 1.3 words adjustment, aligned with JS worker)."""
    if not text:
        return 0
    trimmed = text.strip()
    if not trimmed:
        return 0
    words = len(trimmed.split())
    chars = len(trimmed)
    return max(1, round((chars / 3.8 + words * 1.3) / 2))
