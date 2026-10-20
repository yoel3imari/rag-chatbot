"""Server-Sent Event encoding for ingestion progress."""

import json
from typing import Any, Dict


def sse_event(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"
