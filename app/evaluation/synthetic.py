"""Synthetic evaluation case generation from indexed chunks."""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import KEYLESS_PROVIDERS
from app.evaluation.datasets import ensure_default_handbook_seeded
from app.store import store
from app.vector_store import vector_store


def _extract_facts_from_chunk(chunk_text: str) -> List[str]:
    """Extracts salient factual tokens (numbers, time ranges, paths, acronyms) from a chunk."""
    facts: List[str] = []
    patterns = [
        r"\$\d[\d,]*",
        r"\b\d{1,2}:\d{2}\s*UTC\b",
        r"\b\d+(?:-|\s+)(?:days?|weeks?|hours?|minutes?|years?|months?)\b",
        r"\b[A-Z]{2,6}-\d+(?:-[A-Z0-9]+)*\b",
        r"`[^`]{2,40}`",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, chunk_text):
            value = match.group(0).strip("`").strip()
            if value and value not in facts:
                facts.append(value)
    if len(facts) < 2:
        words = [
            word.strip(".,:;()[]\"'*#")
            for word in chunk_text.split()
            if len(word.strip(".,:;()[]\"'*#")) >= 5
        ]
        stop_words = {
            "which",
            "their",
            "there",
            "these",
            "those",
            "where",
            "while",
            "after",
            "before",
            "under",
            "about",
            "through",
            "between",
            "within",
            "without",
        }
        for word in words:
            if word.lower() not in stop_words and word not in facts:
                facts.append(word)
            if len(facts) >= 3:
                break
    return facts[:4]


async def generate_synthetic_eval_cases(
    doc_ids: Optional[List[str]] = None,
    max_cases: int = 6,
    use_llm: bool = False,
    user_id: str = "default",
) -> List[Dict[str, Any]]:
    """
    Generates synthetic Q&A cases directly from chunks stored in LanceDB.
    Supports deterministic extraction and optional LLM-assisted generation.
    """
    ensure_default_handbook_seeded()
    try:
        all_rows = vector_store.table.to_arrow().to_pylist()
    except Exception:
        all_rows = []

    if doc_ids:
        document_set = set(doc_ids)
        all_rows = [
            row
            for row in all_rows
            if row.get("doc_id") in document_set
            or row.get("source_name") in document_set
        ]

    all_rows = [
        row for row in all_rows if row.get("user_id", "default") == user_id
    ]
    if not all_rows:
        return []

    settings = store.get_settings(user_id=user_id, include_decrypted_key=True)
    api_key = settings.get("_decrypted_api_key", "")
    provider = settings.get("provider", "openrouter")
    can_use_llm = use_llm and (bool(api_key) or provider in KEYLESS_PROVIDERS)

    synthetic_cases: List[Dict[str, Any]] = []
    for index, row in enumerate(all_rows[: max_cases * 2]):
        if len(synthetic_cases) >= max_cases:
            break
        raw_text = (row.get("text") or "").strip()
        if len(raw_text) < 40:
            continue
        source_name = row.get("source_name") or "document.md"
        chunk_index = int(row.get("chunk_index", index))

        heading_match = re.search(r"^#{1,4}\s+(.+)$", raw_text, flags=re.MULTILINE)
        heading = heading_match.group(1).strip() if heading_match else ""
        clean_topic = (
            re.sub(r"^\d+[\.\)]\s*", "", heading).strip() if heading else ""
        )

        facts = _extract_facts_from_chunk(raw_text)
        body_lines = [
            line.strip().lstrip("-* ").strip()
            for line in raw_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        summary_sentence = " ".join(body_lines)[:260].strip()
        if not summary_sentence:
            summary_sentence = raw_text[:240]

        question = ""
        expected_answer = f"{summary_sentence} [{source_name}]"

        if can_use_llm:
            try:
                llm = ChatOpenAI(
                    api_key=api_key or "not-needed",
                    base_url=settings.get(
                        "base_url", "https://openrouter.ai/api/v1"
                    ),
                    model=settings.get("model", "openai/gpt-4o-mini"),
                    temperature=0.2,
                    max_tokens=220,
                    timeout=15.0,
                    max_retries=1,
                )
                prompt = (
                    "Given the following knowledge base chunk, generate ONE specific factual question "
                    "that can be answered directly from the chunk, and a concise ground-truth answer.\n"
                    'Respond strictly in JSON: {"question": "...", "expected_answer": "...", "expected_facts": ["fact1", "fact2"]}\n\n'
                    f"Source: {source_name}\nChunk:\n{raw_text}"
                )
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                content = str(response.content or "").strip()
                json_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    question = str(parsed.get("question") or "").strip()
                    if parsed.get("expected_answer"):
                        expected_answer = (
                            f"{parsed['expected_answer'].strip()} [{source_name}]"
                        )
                    if (
                        isinstance(parsed.get("expected_facts"), list)
                        and parsed["expected_facts"]
                    ):
                        facts = [str(value) for value in parsed["expected_facts"][:4]]
            except Exception:
                question = ""

        if not question:
            if clean_topic:
                question = (
                    f"According to {source_name}, what are the policies or details "
                    f"regarding {clean_topic}?"
                )
            else:
                first_clause = re.split(r"[.!?\n]", summary_sentence)[0].strip()
                snippet_words = " ".join(first_clause.split()[:7])
                question = (
                    f"What does {source_name} state regarding '{snippet_words}'?"
                )

        synthetic_cases.append(
            {
                "id": f"synth-{index + 1:02d}",
                "category": "synthetic",
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": [source_name],
                "expected_facts": facts,
                "target_docs": [],
                "expect_retrieval": True,
                "allow_abstain": False,
                "source_chunk_index": chunk_index,
            }
        )

    return synthetic_cases
