import argparse
import asyncio
import json
import math
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent import (
    RETRIEVAL_LC_TOOLS,
    _format_rag_context_block,
    _is_fatal_upstream_error,
    _safe_hybrid_search,
    _should_auto_retrieve_heuristic,
)
from app.config import EMBEDDING_MODELS, KEYLESS_PROVIDERS
from app.embeddings import embedding_engine
from app.ingestion import chunk_text
from app.models import EvalCaseModel
from app.store import estimate_tokens, store
from app.vector_store import vector_store


SAMPLE_COMPANY_HANDBOOK_TEXT = """# ACME CORP — ENGINEERING & OPERATIONS HANDBOOK (2026)

## 1. Remote Work & Core Hours
Acme Corp operates on a hybrid-first engineering model. Core synchronous collaboration hours are 13:00 UTC to 17:00 UTC Monday through Friday.
All full-time employees receive an annual $2,500 Home Office Stipend and $1,200 Learning & Conference budget.

## 2. Paid Time Off (PTO) & Sabbatical Policy
Employees accrue 25 days of Paid Time Off (PTO) per calendar year, plus 12 company holidays.
After 4 years of continuous tenure, engineers qualify for a 4-week paid sabbatical.

## 3. Production Deployment & Incident Response
- **Deployment Freeze:** No routine production deployments on Fridays after 15:00 UTC without VP of Engineering sign-off.
- **SEV-1 SLA:** Critical customer-facing outages require initial on-call acknowledgement via PagerDuty within 5 minutes and a public status page update within 15 minutes.
- **Postmortems:** Blameless postmortems must be published in the `/incidents` repository within 48 hours of resolution.

## 4. Security & Credential Hygiene
All API keys and production secrets must be encrypted at rest using AES-256-GCM or stored in Vault. Rotating service tokens every 90 days is mandatory.
"""


GOLDEN_EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "id": "golden-01",
        "category": "rag_factual",
        "question": "What are Acme Corp's core synchronous collaboration hours and annual employee stipends?",
        "expected_answer": (
            "Acme Corp's core synchronous collaboration hours are 13:00 UTC to 17:00 UTC Monday through Friday. "
            "Full-time employees receive an annual $2,500 Home Office Stipend and a $1,200 Learning & Conference budget [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["13:00 UTC", "17:00 UTC", "$2,500", "$1,200"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-02",
        "category": "rag_factual",
        "question": "How many days of Paid Time Off (PTO) do employees accrue, and when do engineers qualify for a sabbatical?",
        "expected_answer": (
            "Employees accrue 25 days of Paid Time Off (PTO) per calendar year plus 12 company holidays. "
            "After 4 years of continuous tenure, engineers qualify for a 4-week paid sabbatical [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["25 days", "12 company holidays", "4 years", "4-week"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-03",
        "category": "rag_factual",
        "question": "What is the Friday production deployment freeze rule and the SEV-1 outage response SLA?",
        "expected_answer": (
            "Routine production deployments are frozen on Fridays after 15:00 UTC without VP of Engineering sign-off. "
            "SEV-1 critical outages require initial on-call acknowledgement via PagerDuty within 5 minutes and a public status page update within 15 minutes [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["15:00 UTC", "VP of Engineering", "5 minutes", "15 minutes"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-04",
        "category": "rag_factual",
        "question": "Where and within what timeframe must blameless incident postmortems be published?",
        "expected_answer": (
            "Blameless postmortems must be published in the `/incidents` repository within 48 hours of resolution [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["/incidents", "48 hours"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-05",
        "category": "targeted_doc",
        "question": "What encryption standard and service token rotation schedule are required for production secrets?",
        "expected_answer": (
            "All API keys and production secrets must be encrypted at rest using AES-256-GCM or stored in Vault, "
            "and rotating service tokens every 90 days is mandatory [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["AES-256-GCM", "Vault", "90 days"],
        "target_docs": ["acme-handbook-2026.md"],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-06",
        "category": "unanswerable",
        "question": "What is Acme Corp's corporate helicopter reimbursement limit and pet insurance deductible?",
        "expected_answer": (
            "The knowledge base does not contain any policy or information regarding corporate helicopter reimbursement or pet insurance deductibles."
        ),
        "expected_sources": [],
        "expected_facts": [],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": True,
    },
    {
        "id": "golden-07",
        "category": "chitchat",
        "question": "Hello! Good morning!",
        "expected_answer": "Hello! How can I help you today?",
        "expected_sources": [],
        "expected_facts": [],
        "target_docs": [],
        "expect_retrieval": False,
        "allow_abstain": False,
    },
    {
        "id": "golden-08",
        "category": "math",
        "question": "25 * 4 + 50 = ?",
        "expected_answer": "150",
        "expected_sources": [],
        "expected_facts": ["150"],
        "target_docs": [],
        "expect_retrieval": False,
        "allow_abstain": False,
    },
]


def ensure_default_handbook_seeded() -> None:
    """Ensures the default Acme Corp handbook is indexed in SQLite and LanceDB before running evaluations."""
    docs = store.list_documents()
    has_handbook = any(d.get("name") == "acme-handbook-2026.md" for d in docs)
    try:
        row_count = vector_store.table.count_rows()
    except Exception:
        row_count = 0

    if not has_handbook or row_count == 0:
        clean_handbook = SAMPLE_COMPANY_HANDBOOK_TEXT.replace(
            "Setelah 4 years of continuous tenure, engineers qualify for a 4-week paid sabbatical.\n", ""
        )
        doc = store.create_document_record(
            name="acme-handbook-2026.md",
            source_type="file",
            scope="global",
            file_size=len(clean_handbook.encode("utf-8")),
        )
        doc_id = doc["id"]
        chunks = chunk_text(clean_handbook)
        vectors = embedding_engine.embed_batch(chunks)
        records = [
            {
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source_name": "acme-handbook-2026.md",
                "source_type": "file",
                "scope": "global",
                "session_id": "",
                "user_id": "default",
                "embedding_model": embedding_engine.model_name,
                "chunk_index": i,
                "text": c,
                "vector": v,
            }
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        vector_store.add_chunks(records)
        store.update_document_status(doc_id, "INDEXED", chunk_count=len(chunks))


def _cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a / na, b / nb), 0.0, 1.0))


def _fact_present(fact: str, text: str) -> bool:
    """Checks if an expected fact or key phrase is present in text (case-insensitive, whitespace-normalized)."""
    if not fact or not text:
        return False
    norm_fact = re.sub(r"\s+", " ", fact.strip().lower())
    norm_text = re.sub(r"\s+", " ", text.strip().lower())
    if norm_fact in norm_text:
        return True
    # Also check stripped punctuation match (e.g., "$2,500" vs "2500" or "4-week" vs "4 week")
    alnum_fact = re.sub(r"[^a-z0-9/]+", " ", norm_fact).strip()
    alnum_text = re.sub(r"[^a-z0-9/]+", " ", norm_text).strip()
    if alnum_fact and len(alnum_fact) >= 2 and alnum_fact in alnum_text:
        return True
    return False


def _extract_facts_from_chunk(chunk_text_str: str) -> List[str]:
    """Extracts salient factual tokens (numbers with units, time ranges, paths, acronyms, key phrases) from a chunk."""
    facts: List[str] = []
    # 1. Currency, time, durations, percentages, or hyphenated specs (e.g., AES-256-GCM, SEV-1, 13:00 UTC, $2,500, 48 hours)
    patterns = [
        r"\$\d[\d,]*",
        r"\b\d{1,2}:\d{2}\s*UTC\b",
        r"\b\d+(?:-|\s+)(?:days?|weeks?|hours?|minutes?|years?|months?)\b",
        r"\b[A-Z]{2,6}-\d+(?:-[A-Z0-9]+)*\b",
        r"`[^`]{2,40}`",
    ]
    for pat in patterns:
        for m in re.finditer(pat, chunk_text_str):
            val = m.group(0).strip("`").strip()
            if val and val not in facts:
                facts.append(val)
    if len(facts) < 2:
        # Extract meaningful noun/keyword phrases from sentences
        words = [
            w.strip(".,:;()[]\"'*#")
            for w in chunk_text_str.split()
            if len(w.strip(".,:;()[]\"'*#")) >= 5
        ]
        stop = {
            "which", "their", "there", "these", "those", "where", "while", "after",
            "before", "under", "about", "through", "between", "within", "without",
        }
        for w in words:
            if w.lower() not in stop and w not in facts:
                facts.append(w)
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
    Generates synthetic evaluation Q&A cases directly from chunks stored in LanceDB.
    Supports both deterministic rule-based extraction (zero external API calls) and LLM-assisted generation.
    """
    ensure_default_handbook_seeded()
    try:
        all_rows = vector_store.table.to_arrow().to_pylist()
    except Exception:
        all_rows = []

    if doc_ids:
        doc_set = set(doc_ids)
        all_rows = [
            r for r in all_rows if r.get("doc_id") in doc_set or r.get("source_name") in doc_set
        ]

    # Filter to user_id
    all_rows = [r for r in all_rows if r.get("user_id", "default") == user_id]
    if not all_rows:
        return []

    settings = store.get_settings(user_id=user_id, include_decrypted_key=True)
    api_key = settings.get("_decrypted_api_key", "")
    provider = settings.get("provider", "openrouter")
    can_use_llm = use_llm and (bool(api_key) or provider in KEYLESS_PROVIDERS)

    synthetic_cases: List[Dict[str, Any]] = []
    for idx, row in enumerate(all_rows[: max_cases * 2]):
        if len(synthetic_cases) >= max_cases:
            break
        raw_text = (row.get("text") or "").strip()
        if len(raw_text) < 40:
            continue
        source_name = row.get("source_name") or "document.md"
        chunk_idx = int(row.get("chunk_index", idx))

        # Extract heading if present
        heading_match = re.search(r"^#{1,4}\s+(.+)$", raw_text, flags=re.MULTILINE)
        heading = heading_match.group(1).strip() if heading_match else ""
        # Strip leading section number like "1. "
        clean_topic = re.sub(r"^\d+[\.\)]\s*", "", heading).strip() if heading else ""

        facts = _extract_facts_from_chunk(raw_text)
        body_lines = [
            ln.strip().lstrip("-* ").strip()
            for ln in raw_text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
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
                    base_url=settings.get("base_url", "https://openrouter.ai/api/v1"),
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
                resp = await llm.ainvoke([HumanMessage(content=prompt)])
                content_str = str(resp.content or "").strip()
                json_match = re.search(r"\{.*\}", content_str, flags=re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    question = str(parsed.get("question") or "").strip()
                    if parsed.get("expected_answer"):
                        expected_answer = f"{parsed['expected_answer'].strip()} [{source_name}]"
                    if isinstance(parsed.get("expected_facts"), list) and parsed["expected_facts"]:
                        facts = [str(x) for x in parsed["expected_facts"][:4]]
            except Exception:
                question = ""

        if not question:
            if clean_topic:
                question = f"According to {source_name}, what are the policies or details regarding {clean_topic}?"
            else:
                first_clause = re.split(r"[.!?\n]", summary_sentence)[0].strip()
                snippet_words = " ".join(first_clause.split()[:7])
                question = f"What does {source_name} state regarding '{snippet_words}'?"

        synthetic_cases.append(
            {
                "id": f"synth-{idx + 1:02d}",
                "category": "synthetic",
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": [source_name],
                "expected_facts": facts,
                "target_docs": [],
                "expect_retrieval": True,
                "allow_abstain": False,
                "source_chunk_index": chunk_idx,
            }
        )

    return synthetic_cases


def evaluate_retrieval_metrics(
    case: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    rag_token_ceiling: int = 1500,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes retrieval quality metrics for a single test case:
    - hit_rate (Recall@K binary hit)
    - mrr (Mean Reciprocal Rank)
    - ndcg (Normalized Discounted Cumulative Gain @ K)
    - context_precision (fraction of retrieved chunks that are relevant)
    - context_recall (fraction of expected ground-truth facts present in retrieved chunks)
    - rag_tokens & budget_compliant
    """
    expect_retrieval = bool(case.get("expect_retrieval", True))
    allow_abstain = bool(case.get("allow_abstain", False))
    expected_sources = [s.lower() for s in (case.get("expected_sources") or []) if s]
    expected_facts = [f for f in (case.get("expected_facts") or []) if f]

    formatted_block, citations = _format_rag_context_block(retrieved_chunks, rag_token_ceiling)
    rag_tokens = estimate_tokens(formatted_block) if retrieved_chunks else 0
    budget_compliant = rag_tokens <= rag_token_ceiling

    # For non-retrieval cases (chitchat / math)
    if not expect_retrieval:
        return {
            "hit_rate": 1.0 if len(retrieved_chunks) == 0 else 0.5,
            "mrr": 1.0 if len(retrieved_chunks) == 0 else 0.5,
            "ndcg": 1.0 if len(retrieved_chunks) == 0 else 0.5,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "first_relevant_rank": None,
            "relevant_chunks_count": 0,
            "retrieved_chunks_count": len(retrieved_chunks),
            "rag_tokens": rag_tokens,
            "budget_compliant": budget_compliant,
            "matched_facts": expected_facts,
            "missing_facts": [],
        }

    # For unanswerable / negative queries where expected_sources and expected_facts are empty
    if allow_abstain and not expected_sources and not expected_facts:
        # Check that top similarity isn't falsely high or that system handled negative query cleanly
        max_sim = max((float(c.get("similarity", 0.0)) for c in retrieved_chunks), default=0.0)
        score = 1.0 if max_sim < 0.65 else 0.7
        return {
            "hit_rate": score,
            "mrr": score,
            "ndcg": score,
            "context_precision": score,
            "context_recall": 1.0,
            "first_relevant_rank": None,
            "relevant_chunks_count": 0,
            "retrieved_chunks_count": len(retrieved_chunks),
            "rag_tokens": rag_tokens,
            "budget_compliant": budget_compliant,
            "matched_facts": [],
            "missing_facts": [],
        }

    if not retrieved_chunks:
        return {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "first_relevant_rank": None,
            "relevant_chunks_count": 0,
            "retrieved_chunks_count": 0,
            "rag_tokens": 0,
            "budget_compliant": True,
            "matched_facts": [],
            "missing_facts": expected_facts,
        }

    # Evaluate per-chunk relevance grade (0.0 to 1.0)
    chunk_grades: List[float] = []
    first_relevant_rank: Optional[int] = None
    relevant_count = 0

    for rank_idx, chunk in enumerate(retrieved_chunks):
        c_source = (chunk.get("source_name") or "").lower()
        c_doc_id = (chunk.get("doc_id") or "").lower()
        c_text = chunk.get("text") or ""

        source_ok = True
        if expected_sources:
            source_ok = (c_source in expected_sources) or (c_doc_id in expected_sources)

        if expected_facts:
            fact_hits = sum(1 for f in expected_facts if _fact_present(f, c_text))
            fact_ratio = fact_hits / float(len(expected_facts))
        else:
            fact_ratio = 1.0 if float(chunk.get("similarity", 0.0)) >= 0.25 else 0.0

        # A chunk is relevant if it matches the expected source and contains at least one expected fact (or high semantic similarity)
        if source_ok and (fact_ratio > 0.0 or float(chunk.get("similarity", 0.0)) >= 0.35):
            grade = 0.4 + 0.6 * fact_ratio
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank_idx + 1
        elif fact_ratio > 0.0:
            grade = 0.5 * fact_ratio
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank_idx + 1
        else:
            grade = 0.0

        chunk_grades.append(grade)

    hit_rate = 1.0 if first_relevant_rank is not None else 0.0
    mrr = round(1.0 / first_relevant_rank, 4) if first_relevant_rank is not None else 0.0

    # Compute NDCG@K
    dcg = sum(g / math.log2(idx + 2) for idx, g in enumerate(chunk_grades))
    ideal_grades = sorted(chunk_grades, reverse=True)
    if max(ideal_grades, default=0.0) < 1.0 and hit_rate > 0:
        ideal_grades[0] = 1.0
    idcg = sum(g / math.log2(idx + 2) for idx, g in enumerate(ideal_grades))
    ndcg = round(min(1.0, dcg / idcg), 4) if idcg > 0 else 0.0

    context_precision = round(relevant_count / float(len(retrieved_chunks)), 4)

    combined_text = "\n".join(c.get("text", "") for c in retrieved_chunks)
    if expected_facts:
        matched_facts = [f for f in expected_facts if _fact_present(f, combined_text)]
        missing_facts = [f for f in expected_facts if f not in matched_facts]
        context_recall = round(len(matched_facts) / float(len(expected_facts)), 4)
    else:
        matched_facts = []
        missing_facts = []
        expected_ans = case.get("expected_answer") or case.get("question") or ""
        vecs = embedding_engine.embed_batch([expected_ans, combined_text[:1500]], model_name=embedding_model)
        context_recall = round(_cosine_sim(vecs[0], vecs[1]), 4) if len(vecs) == 2 else hit_rate

    return {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "ndcg": ndcg,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "first_relevant_rank": first_relevant_rank,
        "relevant_chunks_count": relevant_count,
        "retrieved_chunks_count": len(retrieved_chunks),
        "rag_tokens": rag_tokens,
        "budget_compliant": budget_compliant,
        "matched_facts": matched_facts,
        "missing_facts": missing_facts,
    }


def _synthesize_deterministic_agent_answer(
    case: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    did_retrieve: bool,
) -> str:
    """
    Synthesizes a deterministic, grounded answer when running in offline/local mode
    (without requiring an external LLM API key).
    """
    question = (case.get("question") or "").strip()
    category = case.get("category", "rag_factual")
    allow_abstain = bool(case.get("allow_abstain", False))

    if not did_retrieve:
        # Check if math expression
        math_clean = re.sub(r"[^0-9\+\-\*\/\(\)\.]", "", question)
        if math_clean and any(ch.isdigit() for ch in math_clean):
            try:
                val = eval(math_clean, {"__builtins__": {}}, {})
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                return f"The result of `{math_clean}` is **{val}**."
            except Exception:
                pass
        return "Hello! How can I assist you with your knowledge base today?"

    if not retrieved_chunks:
        return (
            "I searched the knowledge base, but the retrieved context does not contain "
            "information to answer this question."
        )

    # Check if top chunk similarity is very low or query has zero keyword overlap with chunks (unanswerable detection)
    q_words = {
        w.lower().strip("?.,!\"'()[]")
        for w in question.split()
        if len(w.strip("?.,!\"'()[]")) >= 4
        and w.lower() not in {"what", "when", "where", "which", "does", "acme", "corp", "corps", "policy", "rules", "regarding", "about", "their"}
    }
    combined_lower = " ".join(c.get("text", "").lower() for c in retrieved_chunks)
    keyword_hits = sum(1 for w in q_words if w in combined_lower)
    top_sim = max((float(c.get("similarity", 0.0)) for c in retrieved_chunks), default=0.0)

    if (allow_abstain or category == "unanswerable") and keyword_hits == 0 and top_sim < 0.45:
        return (
            "The retrieved knowledge base context does not contain information regarding "
            f"this topic ({question})."
        )

    # Select sentences from retrieved chunks that best match the question
    selected_excerpts: List[str] = []
    sources_cited: List[str] = []
    for c in retrieved_chunks[:3]:
        src = c.get("source_name") or "document"
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        selected_excerpts.append(f"{txt} [{src}]")
        if src not in sources_cited:
            sources_cited.append(src)

    return "\n\n".join(selected_excerpts)


async def evaluate_generation_metrics(
    case: Dict[str, Any],
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    did_retrieve: bool,
    embedding_model: Optional[str] = None,
    use_llm_judge: bool = False,
    llm_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes generation & agent quality metrics:
    - tool_routing_score: Did the ReAct agent make the expected retrieval vs. direct decision?
    - faithfulness: Is the answer grounded in the retrieved chunks without hallucination?
    - answer_relevance: Does the answer address the question and cover expected facts?
    - citation_accuracy: Are sources properly cited in brackets [source] when retrieval was used?
    """
    expect_retrieval = bool(case.get("expect_retrieval", True))
    allow_abstain = bool(case.get("allow_abstain", False))
    expected_sources = [s.lower() for s in (case.get("expected_sources") or []) if s]
    expected_facts = [f for f in (case.get("expected_facts") or []) if f]
    expected_answer = (case.get("expected_answer") or "").strip()
    question = (case.get("question") or "").strip()

    # 1. ReAct Tool Routing Accuracy
    tool_routing_score = 1.0 if (did_retrieve == expect_retrieval) else 0.0

    if not answer or not answer.strip():
        return {
            "tool_routing_score": tool_routing_score,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "citation_accuracy": 0.0,
            "answer_fact_coverage": 0.0,
            "judge_method": "heuristic",
            "judge_rationale": "Empty response generated.",
        }

    ans_lower = answer.lower()

    # 2. Citation Accuracy
    bracket_citations = [
        m.group(1).strip().lower()
        for m in re.finditer(r"\[([^\[\]]+\.(?:md|pdf|txt|html)|https?://[^\[\]]+)\]", answer, flags=re.IGNORECASE)
    ]
    if not expect_retrieval:
        citation_accuracy = 1.0 if len(bracket_citations) == 0 else 0.5
    elif allow_abstain and not expected_sources:
        citation_accuracy = 1.0
    elif expected_sources:
        has_expected_cit = any(
            any(exp_src in bc or exp_src in ans_lower for exp_src in expected_sources)
            for bc in (bracket_citations or [ans_lower])
        )
        if bracket_citations and any(bc in expected_sources for bc in bracket_citations):
            citation_accuracy = 1.0
        elif has_expected_cit:
            citation_accuracy = 0.85
        else:
            citation_accuracy = 0.0
    else:
        citation_accuracy = 1.0 if bracket_citations else 0.7

    # 3. Answer Fact Coverage
    if expected_facts:
        ans_fact_hits = sum(1 for f in expected_facts if _fact_present(f, answer))
        answer_fact_coverage = round(ans_fact_hits / float(len(expected_facts)), 4)
    else:
        answer_fact_coverage = 1.0

    # 4. Abstention / Unanswerable handling
    abstain_phrases = (
        "does not contain",
        "no matching documents",
        "not mentioned",
        "no information",
        "cannot find",
        "not specify",
        "not in the knowledge base",
    )
    did_abstain = any(p in ans_lower for p in abstain_phrases)

    # 5. Deterministic Semantic + Lexical Faithfulness & Relevance
    combined_context = "\n\n".join(c.get("text", "") for c in retrieved_chunks)
    if not expect_retrieval:
        # For chitchat / math
        faithfulness = 1.0
        if expected_facts:
            answer_relevance = answer_fact_coverage
        else:
            vecs = embedding_engine.embed_batch(
                [expected_answer or question, answer[:1000]], model_name=embedding_model
            )
            sim = _cosine_sim(vecs[0], vecs[1]) if len(vecs) == 2 else 0.8
            answer_relevance = round(max(0.75, sim), 4)
    elif allow_abstain and not expected_facts:
        if did_abstain:
            faithfulness = 1.0
            answer_relevance = 1.0
        else:
            faithfulness = 0.35
            answer_relevance = 0.4
    else:
        if not retrieved_chunks:
            faithfulness = 1.0 if did_abstain else 0.0
            answer_relevance = 0.0
        else:
            # Check token/number grounding in context (strip [source] citations first so numbers in filenames aren't penalized)
            answer_without_citations = re.sub(r"\[[^\[\]]+\]", " ", answer)
            ans_numbers = set(re.findall(r"\b\d[\d,.:]*\b", answer_without_citations))
            ctx_numbers = set(re.findall(r"\b\d[\d,.:]*\b", combined_context))
            # Ignore small citation indices like 1, 2, 3
            significant_ans_nums = {n for n in ans_numbers if len(n) >= 2 or int(re.sub(r"\D", "", n) or "0") > 4}
            if significant_ans_nums:
                grounded_nums = sum(1 for n in significant_ans_nums if n in ctx_numbers or n in combined_context)
                num_grounding = grounded_nums / float(len(significant_ans_nums))
            else:
                num_grounding = 1.0

            vecs = embedding_engine.embed_batch(
                [combined_context[:1600], answer[:1200], expected_answer or question],
                model_name=embedding_model,
            )
            ctx_ans_sim = _cosine_sim(vecs[0], vecs[1]) if len(vecs) == 3 else 0.7
            exp_ans_sim = _cosine_sim(vecs[2], vecs[1]) if len(vecs) == 3 else 0.7

            faithfulness = round(min(1.0, 0.55 * num_grounding + 0.45 * min(1.0, ctx_ans_sim * 1.25)), 4)
            if expected_facts:
                answer_relevance = round(0.65 * answer_fact_coverage + 0.35 * min(1.0, exp_ans_sim * 1.2), 4)
            else:
                answer_relevance = round(min(1.0, exp_ans_sim * 1.2), 4)

    judge_method = "semantic_heuristic"
    judge_rationale = (
        f"Fact coverage: {int(answer_fact_coverage * 100)}%, "
        f"Tool routing: {'PASS' if tool_routing_score == 1.0 else 'FAIL'}, "
        f"Citation score: {int(citation_accuracy * 100)}%."
    )

    # 6. Optional LLM-as-a-Judge Refinement
    if use_llm_judge and llm_settings:
        api_key = llm_settings.get("_decrypted_api_key", "")
        provider = llm_settings.get("provider", "openrouter")
        if api_key or provider in KEYLESS_PROVIDERS:
            try:
                llm = ChatOpenAI(
                    api_key=api_key or "not-needed",
                    base_url=llm_settings.get("base_url", "https://openrouter.ai/api/v1"),
                    model=llm_settings.get("model", "openai/gpt-4o-mini"),
                    temperature=0.0,
                    max_tokens=240,
                    timeout=20.0,
                    max_retries=1,
                )
                judge_prompt = (
                    "You are an impartial RAG evaluation judge. Score the assistant's response on two metrics from 0.0 to 1.0:\n"
                    "1. `faithfulness`: Is the response strictly grounded in the Retrieved Context without hallucinating facts?\n"
                    "2. `answer_relevance`: Does the response directly and accurately answer the Question compared to the Expected Answer?\n"
                    'Respond strictly in JSON: {"faithfulness": 0.95, "answer_relevance": 0.95, "rationale": "Brief 1-sentence explanation"}\n\n'
                    f"Question: {question}\n"
                    f"Expected Answer: {expected_answer}\n"
                    f"Retrieved Context:\n{combined_context[:1800] or '(None)'}\n\n"
                    f"Assistant Response:\n{answer[:1200]}"
                )
                resp = await llm.ainvoke([HumanMessage(content=judge_prompt)])
                raw_judge = str(resp.content or "").strip()
                jm = re.search(r"\{.*\}", raw_judge, flags=re.DOTALL)
                if jm:
                    jdata = json.loads(jm.group(0))
                    if "faithfulness" in jdata:
                        faithfulness = round(
                            0.5 * faithfulness + 0.5 * float(np.clip(float(jdata["faithfulness"]), 0.0, 1.0)),
                            4,
                        )
                    if "answer_relevance" in jdata:
                        answer_relevance = round(
                            0.5 * answer_relevance + 0.5 * float(np.clip(float(jdata["answer_relevance"]), 0.0, 1.0)),
                            4,
                        )
                    if jdata.get("rationale"):
                        judge_rationale = str(jdata["rationale"]).strip()
                    judge_method = f"llm_judge ({llm_settings.get('model')})"
            except Exception:
                pass

    return {
        "tool_routing_score": tool_routing_score,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "citation_accuracy": round(citation_accuracy, 4),
        "answer_fact_coverage": answer_fact_coverage,
        "judge_method": judge_method,
        "judge_rationale": judge_rationale,
    }


async def evaluate_single_case(
    case: Dict[str, Any],
    mode: str = "full",
    top_k: int = 5,
    similarity_threshold: float = 0.15,
    dedup_threshold: float = 0.92,
    rag_token_ceiling: int = 1500,
    embedding_model: str = "all-MiniLM-L6-v2",
    settings: Optional[Dict[str, Any]] = None,
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Runs a single evaluation test case through the RAG retrieval pipeline and ReAct agent,
    returning full retrieval metrics, agent/generation metrics, trace steps, and latency.
    """
    t_start = time.perf_counter()
    question = (case.get("question") or "").strip()
    target_docs = list(case.get("target_docs") or [])
    expect_retrieval = bool(case.get("expect_retrieval", True))
    settings = settings or store.get_settings(user_id=user_id, include_decrypted_key=True)

    api_key = settings.get("_decrypted_api_key", "")
    provider = settings.get("provider", "openrouter")
    has_live_llm = (mode in ("full", "llm_judge")) and (bool(api_key) or provider in KEYLESS_PROVIDERS)

    react_steps: List[Dict[str, Any]] = []
    did_retrieve = False
    search_query = question
    effective_targets = list(target_docs)

    # Step 1: ReAct Tool Routing Decision
    if has_live_llm and mode == "llm_judge":
        try:
            llm = ChatOpenAI(
                api_key=api_key or "not-needed",
                base_url=settings.get("base_url", "https://openrouter.ai/api/v1"),
                model=settings.get("model", "openai/gpt-4o-mini"),
                temperature=0.0,
                max_tokens=300,
                timeout=20.0,
                max_retries=1,
            ).bind_tools(RETRIEVAL_LC_TOOLS)
            prompt_msg = question
            if target_docs:
                prompt_msg = f"[Targeted Document Filter Active: {', '.join(target_docs)}]\nUser Query: {question}"
            decision = await llm.ainvoke(
                [
                    SystemMessage(content=settings.get("system_prompt", "")),
                    HumanMessage(content=prompt_msg),
                ]
            )
            if getattr(decision, "tool_calls", None):
                did_retrieve = True
                tc = decision.tool_calls[0]
                args = tc.get("args") or {}
                search_query = args.get("query") or question
                if args.get("target_source") and args["target_source"] not in effective_targets:
                    effective_targets.append(args["target_source"])
                react_steps.append(
                    {
                        "step": "THOUGHT",
                        "detail": args.get("reasoning") or "LLM decided to invoke search_knowledge_base.",
                    }
                )
            elif target_docs:
                did_retrieve = True
                react_steps.append(
                    {
                        "step": "THOUGHT",
                        "detail": f"Explicit @mention filter for {', '.join(target_docs)}.",
                    }
                )
            else:
                did_retrieve = False
                react_steps.append(
                    {
                        "step": "THOUGHT",
                        "detail": "Direct response selected without knowledge base retrieval.",
                    }
                )
        except Exception as exc:
            if not _is_fatal_upstream_error(exc):
                pass
            did_retrieve, reason = _should_auto_retrieve_heuristic(question, target_docs)
            react_steps.append({"step": "THOUGHT", "detail": reason})
    else:
        did_retrieve, reason = _should_auto_retrieve_heuristic(question, target_docs)
        react_steps.append({"step": "THOUGHT", "detail": reason})

    # Step 2: Execute Hybrid Retrieval (if routed to retrieve, or if mode == 'retrieval' and expect_retrieval)
    t_ret_start = time.perf_counter()
    retrieved_chunks: List[Dict[str, Any]] = []
    if did_retrieve or (mode == "retrieval" and expect_retrieval):
        react_steps.append(
            {
                "step": "ACTION",
                "detail": f'search_knowledge_base(query="{search_query}", target_docs={effective_targets})',
            }
        )
        retrieved_chunks, ret_err = await _safe_hybrid_search(
            query=search_query,
            user_id=user_id,
            session_id=None,
            use_global_docs=True,
            target_docs=effective_targets if effective_targets else None,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            dedup_threshold=dedup_threshold,
            embedding_model=embedding_model,
        )
        if ret_err:
            react_steps.append({"step": "ERROR", "detail": ret_err})
        else:
            react_steps.append(
                {
                    "step": "OBSERVATION",
                    "detail": f"Retrieved {len(retrieved_chunks)} chunks from LanceDB.",
                }
            )
    retrieval_latency_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)

    # Step 3: Compute Retrieval Metrics
    ret_metrics = await asyncio.to_thread(
        evaluate_retrieval_metrics,
        case,
        retrieved_chunks,
        rag_token_ceiling,
        embedding_model,
    )

    # Step 4: Answer Synthesis & Generation Metrics
    generated_answer = ""
    if mode == "retrieval":
        tool_routing_score = 1.0 if (did_retrieve == expect_retrieval) else 0.0
        gen_metrics = {
            "tool_routing_score": tool_routing_score,
            "faithfulness": ret_metrics["hit_rate"],
            "answer_relevance": ret_metrics["context_recall"],
            "citation_accuracy": 1.0,
            "answer_fact_coverage": ret_metrics["context_recall"],
            "judge_method": "retrieval_only",
            "judge_rationale": "Retrieval-only benchmark mode.",
        }
    else:
        if has_live_llm and mode == "llm_judge":
            try:
                rag_context_str, _ = _format_rag_context_block(retrieved_chunks, rag_token_ceiling)
                llm_gen = ChatOpenAI(
                    api_key=api_key or "not-needed",
                    base_url=settings.get("base_url", "https://openrouter.ai/api/v1"),
                    model=settings.get("model", "openai/gpt-4o-mini"),
                    temperature=0.1,
                    max_tokens=400,
                    timeout=25.0,
                    max_retries=1,
                )
                if did_retrieve:
                    msgs = [
                        SystemMessage(content=settings.get("system_prompt", "")),
                        SystemMessage(
                            content=(
                                f"## RETRIEVED KNOWLEDGE BASE CONTEXT\n{rag_context_str}\n\n"
                                "Answer accurately using the retrieved context above and cite sources in brackets like [filename.md]."
                            )
                        ),
                        HumanMessage(content=question),
                    ]
                else:
                    msgs = [
                        SystemMessage(content=settings.get("system_prompt", "")),
                        HumanMessage(content=question),
                    ]
                resp = await llm_gen.ainvoke(msgs)
                generated_answer = str(resp.content or "").strip()
            except Exception:
                generated_answer = ""

        if not generated_answer:
            generated_answer = _synthesize_deterministic_agent_answer(
                case=case,
                retrieved_chunks=retrieved_chunks,
                did_retrieve=did_retrieve,
            )

        gen_metrics = await evaluate_generation_metrics(
            case=case,
            answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            did_retrieve=did_retrieve,
            embedding_model=embedding_model,
            use_llm_judge=(mode == "llm_judge" and has_live_llm),
            llm_settings=settings,
        )

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    # Compute overall case score (weighted average of retrieval & agent metrics)
    if mode == "retrieval":
        case_score = round(
            0.30 * ret_metrics["hit_rate"]
            + 0.25 * ret_metrics["mrr"]
            + 0.25 * ret_metrics["context_recall"]
            + 0.20 * gen_metrics["tool_routing_score"],
            4,
        )
    elif not expect_retrieval:
        case_score = round(
            0.50 * gen_metrics["tool_routing_score"]
            + 0.30 * gen_metrics["answer_relevance"]
            + 0.20 * gen_metrics["faithfulness"],
            4,
        )
    else:
        case_score = round(
            0.20 * ret_metrics["hit_rate"]
            + 0.15 * ret_metrics["mrr"]
            + 0.20 * ret_metrics["context_recall"]
            + 0.15 * gen_metrics["tool_routing_score"]
            + 0.15 * gen_metrics["faithfulness"]
            + 0.15 * gen_metrics["answer_relevance"],
            4,
        )

    passed = case_score >= 0.70

    return {
        "case_id": case.get("id") or str(uuid.uuid4())[:8],
        "category": case.get("category", "rag_factual"),
        "question": question,
        "expected_answer": case.get("expected_answer", ""),
        "expected_sources": case.get("expected_sources", []),
        "expected_facts": case.get("expected_facts", []),
        "target_docs": target_docs,
        "expect_retrieval": expect_retrieval,
        "did_retrieve": did_retrieve,
        "generated_answer": generated_answer,
        "case_score": case_score,
        "passed": passed,
        "retrieval": ret_metrics,
        "generation": gen_metrics,
        "retrieved_chunks": [
            {
                "source_name": c.get("source_name"),
                "chunk_index": c.get("chunk_index"),
                "similarity": c.get("similarity"),
                "hybrid_score": c.get("hybrid_score"),
                "match_type": c.get("match_type"),
                "preview": (c.get("text") or "")[:200] + ("..." if len(c.get("text") or "") > 200 else ""),
            }
            for c in retrieved_chunks
        ],
        "react_steps": react_steps,
        "retrieval_latency_ms": retrieval_latency_ms,
        "total_latency_ms": total_latency_ms,
    }


def _aggregate_eval_results(
    results: List[Dict[str, Any]],
    mode: str,
    dataset_name: str,
    embedding_model: str,
    llm_model: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "id": str(uuid.uuid4()),
            "mode": mode,
            "dataset_name": dataset_name,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "overall_score": 0.0,
            "total_cases": 0,
            "passed_cases": 0,
            "summary": {},
            "results": [],
            "config": config,
            "created_at": time.time(),
        }

    passed_count = sum(1 for r in results if r["passed"])
    retrieval_cases = [r for r in results if r.get("expect_retrieval", True)]
    ret_denom = len(retrieval_cases) or total

    avg_hit_rate = round(
        sum(r["retrieval"]["hit_rate"] for r in (retrieval_cases or results)) / float(ret_denom), 4
    )
    avg_mrr = round(
        sum(r["retrieval"]["mrr"] for r in (retrieval_cases or results)) / float(ret_denom), 4
    )
    avg_ndcg = round(
        sum(r["retrieval"]["ndcg"] for r in (retrieval_cases or results)) / float(ret_denom), 4
    )
    avg_ctx_precision = round(
        sum(r["retrieval"]["context_precision"] for r in (retrieval_cases or results)) / float(ret_denom), 4
    )
    avg_ctx_recall = round(
        sum(r["retrieval"]["context_recall"] for r in (retrieval_cases or results)) / float(ret_denom), 4
    )

    avg_tool_routing = round(
        sum(r["generation"]["tool_routing_score"] for r in results) / float(total), 4
    )
    avg_faithfulness = round(
        sum(r["generation"]["faithfulness"] for r in results) / float(total), 4
    )
    avg_answer_relevance = round(
        sum(r["generation"]["answer_relevance"] for r in results) / float(total), 4
    )
    avg_citation_acc = round(
        sum(r["generation"]["citation_accuracy"] for r in results) / float(total), 4
    )
    budget_compliance_rate = round(
        sum(1.0 if r["retrieval"]["budget_compliant"] else 0.0 for r in results) / float(total), 4
    )

    avg_ret_latency = round(
        sum(r["retrieval_latency_ms"] for r in results) / float(total), 2
    )
    avg_total_latency = round(
        sum(r["total_latency_ms"] for r in results) / float(total), 2
    )
    overall_score = round(sum(r["case_score"] for r in results) / float(total), 4)

    # Per-category breakdown
    categories: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "passed": 0, "score_sum": 0.0}
        categories[cat]["count"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
        categories[cat]["score_sum"] += r["case_score"]

    category_breakdown = {
        cat: {
            "count": info["count"],
            "passed": info["passed"],
            "avg_score": round(info["score_sum"] / float(info["count"]), 4),
        }
        for cat, info in categories.items()
    }

    summary = {
        "overall_score": overall_score,
        "pass_rate": round(passed_count / float(total), 4),
        "hit_rate": avg_hit_rate,
        "mrr": avg_mrr,
        "ndcg": avg_ndcg,
        "context_precision": avg_ctx_precision,
        "context_recall": avg_ctx_recall,
        "tool_routing_accuracy": avg_tool_routing,
        "faithfulness": avg_faithfulness,
        "answer_relevance": avg_answer_relevance,
        "citation_accuracy": avg_citation_acc,
        "budget_compliance_rate": budget_compliance_rate,
        "avg_retrieval_latency_ms": avg_ret_latency,
        "avg_total_latency_ms": avg_total_latency,
        "category_breakdown": category_breakdown,
    }

    return {
        "id": str(uuid.uuid4()),
        "mode": mode,
        "dataset_name": dataset_name,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "overall_score": overall_score,
        "total_cases": total,
        "passed_cases": passed_count,
        "summary": summary,
        "results": results,
        "config": config,
        "created_at": time.time(),
    }


async def resolve_eval_dataset(
    dataset_source: str = "golden",
    custom_cases: Optional[List[Any]] = None,
    max_synthetic_cases: int = 6,
    user_id: str = "default",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Resolves the requested evaluation dataset ('golden', 'synthetic', 'combined', or 'custom')."""
    ensure_default_handbook_seeded()

    if dataset_source == "custom" and custom_cases:
        normalized = [
            c.model_dump() if isinstance(c, EvalCaseModel) else dict(c)
            for c in custom_cases
        ]
        return f"Custom Dataset ({len(normalized)} cases)", normalized

    if dataset_source == "synthetic":
        synth = await generate_synthetic_eval_cases(max_cases=max_synthetic_cases, user_id=user_id)
        return f"Synthetic Corpus Suite ({len(synth)} cases)", synth

    if dataset_source == "combined":
        synth = await generate_synthetic_eval_cases(max_cases=max_synthetic_cases, user_id=user_id)
        combined = list(GOLDEN_EVAL_DATASET) + synth
        return f"Golden + Synthetic Suite ({len(combined)} cases)", combined

    return f"Golden Benchmark Suite ({len(GOLDEN_EVAL_DATASET)} cases)", list(GOLDEN_EVAL_DATASET)


async def run_evaluation(
    mode: str = "full",
    dataset_source: str = "golden",
    custom_cases: Optional[List[Any]] = None,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    dedup_threshold: Optional[float] = None,
    embedding_model: Optional[str] = None,
    max_synthetic_cases: int = 6,
    persist: bool = True,
    user_id: str = "default",
) -> Dict[str, Any]:
    """Runs a complete evaluation suite and persists the resulting scorecard in SQLite."""
    final_run: Dict[str, Any] = {}
    async for event in run_evaluation_stream(
        mode=mode,
        dataset_source=dataset_source,
        custom_cases=custom_cases,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        dedup_threshold=dedup_threshold,
        embedding_model=embedding_model,
        max_synthetic_cases=max_synthetic_cases,
        persist=persist,
        user_id=user_id,
    ):
        if event.get("type") == "eval_complete":
            final_run = event.get("run") or {}
    return final_run


async def run_evaluation_stream(
    mode: str = "full",
    dataset_source: str = "golden",
    custom_cases: Optional[List[Any]] = None,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    dedup_threshold: Optional[float] = None,
    embedding_model: Optional[str] = None,
    max_synthetic_cases: int = 6,
    persist: bool = True,
    user_id: str = "default",
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator yielding real-time evaluation progress dicts:
    - {"type": "eval_start", "total_cases": N, "dataset_name": ...}
    - {"type": "case_result", "index": i, "total_cases": N, "progress": pct, "result": ...}
    - {"type": "eval_complete", "run": ...}
    """
    settings = store.get_settings(user_id=user_id, include_decrypted_key=True)
    eff_top_k = int(top_k if top_k is not None else settings.get("top_k", 5))
    eff_sim = float(
        similarity_threshold
        if similarity_threshold is not None
        else settings.get("similarity_threshold", 0.15)
    )
    eff_dedup = float(
        dedup_threshold
        if dedup_threshold is not None
        else settings.get("dedup_threshold", 0.92)
    )
    eff_emb = str(embedding_model or settings.get("embedding_model", "all-MiniLM-L6-v2"))
    rag_ceiling = int(settings.get("rag_token_ceiling", 1500))
    llm_model = f"{settings.get('provider', 'openrouter')}/{settings.get('model', 'openai/gpt-4o-mini')}"

    # Ensure corpus is indexed with the target embedding model
    await asyncio.to_thread(vector_store.reindex_for_model, eff_emb)

    dataset_name, cases = await resolve_eval_dataset(
        dataset_source=dataset_source,
        custom_cases=custom_cases,
        max_synthetic_cases=max_synthetic_cases,
        user_id=user_id,
    )

    config = {
        "mode": mode,
        "dataset_source": dataset_source,
        "top_k": eff_top_k,
        "similarity_threshold": eff_sim,
        "dedup_threshold": eff_dedup,
        "embedding_model": eff_emb,
        "rag_token_ceiling": rag_ceiling,
    }

    yield {
        "type": "eval_start",
        "dataset_name": dataset_name,
        "total_cases": len(cases),
        "config": config,
    }

    results: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases):
        case_res = await evaluate_single_case(
            case=case,
            mode=mode,
            top_k=eff_top_k,
            similarity_threshold=eff_sim,
            dedup_threshold=eff_dedup,
            rag_token_ceiling=rag_ceiling,
            embedding_model=eff_emb,
            settings=settings,
            user_id=user_id,
        )
        results.append(case_res)
        progress = round(((idx + 1) / float(max(1, len(cases)))) * 100)
        yield {
            "type": "case_result",
            "index": idx + 1,
            "total_cases": len(cases),
            "progress": progress,
            "result": case_res,
        }

    run_record = _aggregate_eval_results(
        results=results,
        mode=mode,
        dataset_name=dataset_name,
        embedding_model=eff_emb,
        llm_model=llm_model,
        config=config,
    )

    if persist:
        saved = store.save_eval_run(run_record, user_id=user_id)
        if saved:
            run_record = saved

    yield {
        "type": "eval_complete",
        "progress": 100,
        "run": run_record,
    }


def format_eval_markdown_report(run_data: Dict[str, Any]) -> str:
    """Formats an evaluation run dictionary into a clean GitHub-flavored Markdown report."""
    s = run_data.get("summary") or {}
    cfg = run_data.get("config") or {}
    lines = [
        f"# RAG & ReAct Agent Evaluation Report",
        "",
        f"- **Dataset:** `{run_data.get('dataset_name')}`",
        f"- **Mode:** `{run_data.get('mode')}`",
        f"- **Embedding Model:** `{run_data.get('embedding_model')}` (`top_k={cfg.get('top_k')}`, `sim_threshold={cfg.get('similarity_threshold')}`)",
        f"- **Overall Score:** **{s.get('overall_score', 0.0) * 100:.1f}%** ({run_data.get('passed_cases')}/{run_data.get('total_cases')} cases passed)",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Score |",
        "| :--- | :---: |",
        f"| **Retrieval Hit Rate @ K** | `{s.get('hit_rate', 0.0) * 100:.1f}%` |",
        f"| **Mean Reciprocal Rank (MRR)** | `{s.get('mrr', 0.0):.3f}` |",
        f"| **NDCG @ K** | `{s.get('ndcg', 0.0):.3f}` |",
        f"| **Context Precision** | `{s.get('context_precision', 0.0) * 100:.1f}%` |",
        f"| **Context Recall (Fact Coverage)** | `{s.get('context_recall', 0.0) * 100:.1f}%` |",
        f"| **ReAct Tool Routing Accuracy** | `{s.get('tool_routing_accuracy', 0.0) * 100:.1f}%` |",
        f"| **Answer Faithfulness** | `{s.get('faithfulness', 0.0) * 100:.1f}%` |",
        f"| **Answer Relevance** | `{s.get('answer_relevance', 0.0) * 100:.1f}%` |",
        f"| **Citation Accuracy** | `{s.get('citation_accuracy', 0.0) * 100:.1f}%` |",
        f"| **RAG Token Budget Compliance** | `{s.get('budget_compliance_rate', 0.0) * 100:.1f}%` |",
        f"| **Avg Retrieval Latency** | `{s.get('avg_retrieval_latency_ms', 0.0):.1f} ms` |",
        f"| **Avg End-to-End Latency** | `{s.get('avg_total_latency_ms', 0.0):.1f} ms` |",
        "",
        "## Per-Case Breakdown",
        "",
        "| ID | Category | Status | Score | Hit@K | MRR | Recall | Routing | Faithfulness | Relevance | Latency |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in run_data.get("results", []):
        ret = r["retrieval"]
        gen = r["generation"]
        status_str = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"| `{r['case_id']}` | `{r['category']}` | **{status_str}** | `{r['case_score'] * 100:.1f}%` | "
            f"`{ret['hit_rate']:.2f}` | `{ret['mrr']:.2f}` | `{ret['context_recall'] * 100:.0f}%` | "
            f"`{gen['tool_routing_score']:.0f}` | `{gen['faithfulness'] * 100:.0f}%` | "
            f"`{gen['answer_relevance'] * 100:.0f}%` | `{r['total_latency_ms']:.1f}ms` |"
        )
    return "\n".join(lines) + "\n"


def _print_cli_report(run_data: Dict[str, Any]) -> None:
    s = run_data.get("summary") or {}
    cfg = run_data.get("config") or {}
    sep = "=" * 96
    thin = "-" * 96
    print(sep)
    print(f" RAG & REACT AGENT EVALUATION SCORECARD // MODE: {run_data.get('mode', 'full').upper()}")
    print(sep)
    print(
        f" Dataset         : {run_data.get('dataset_name')}\n"
        f" Embedding Model : {run_data.get('embedding_model')} (top_k={cfg.get('top_k')}, sim_threshold={cfg.get('similarity_threshold')})\n"
        f" Overall Score   : {s.get('overall_score', 0.0) * 100:.1f}%  [{run_data.get('passed_cases')}/{run_data.get('total_cases')} PASSED]"
    )
    print(thin)
    print(
        f"  Hit Rate @ K   : {s.get('hit_rate', 0.0) * 100:6.1f}%   |  ReAct Tool Routing : {s.get('tool_routing_accuracy', 0.0) * 100:6.1f}%\n"
        f"  MRR @ K        : {s.get('mrr', 0.0):6.3f}    |  Faithfulness       : {s.get('faithfulness', 0.0) * 100:6.1f}%\n"
        f"  NDCG @ K       : {s.get('ndcg', 0.0):6.3f}    |  Answer Relevance   : {s.get('answer_relevance', 0.0) * 100:6.1f}%\n"
        f"  Context Recall : {s.get('context_recall', 0.0) * 100:6.1f}%   |  Citation Accuracy  : {s.get('citation_accuracy', 0.0) * 100:6.1f}%\n"
        f"  Ctx Precision  : {s.get('context_precision', 0.0) * 100:6.1f}%   |  Avg Latency        : {s.get('avg_total_latency_ms', 0.0):6.1f} ms"
    )
    print(thin)
    print(
        f" {'ID':<10} {'CATEGORY':<14} {'STATUS':<6} {'SCORE':>7} {'HIT':>5} {'MRR':>6} {'RECALL':>7} {'ROUTE':>6} {'FAITH':>7} {'RELEV':>7} {'LATENCY':>9}"
    )
    print(thin)
    for r in run_data.get("results", []):
        ret = r["retrieval"]
        gen = r["generation"]
        st = "PASS" if r["passed"] else "FAIL"
        print(
            f" {r['case_id']:<10} {r['category']:<14} {st:<6} {r['case_score'] * 100:6.1f}% "
            f"{ret['hit_rate']:5.2f} {ret['mrr']:6.2f} {ret['context_recall'] * 100:6.0f}% "
            f"{gen['tool_routing_score']:6.1f} {gen['faithfulness'] * 100:6.0f}% "
            f"{gen['answer_relevance'] * 100:6.0f}% {r['total_latency_ms']:7.1f}ms"
        )
    print(sep)


def _load_custom_dataset_file(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path_str}")
    raw = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    if isinstance(data, dict) and "cases" in data:
        return list(data["cases"])
    if isinstance(data, list):
        return data
    raise ValueError("Dataset file must contain a JSON array of test cases or an object with a 'cases' key.")


async def _cli_main_async(args: argparse.Namespace) -> int:
    custom_cases = None
    dataset_source = args.dataset_source
    if args.dataset_file:
        custom_cases = _load_custom_dataset_file(args.dataset_file)
        dataset_source = "custom"
    elif args.generate_synthetic:
        dataset_source = "combined"

    if args.compare_models:
        print("=" * 96)
        print(" MULTI-MODEL CRISPEMBED BENCHMARK COMPARISON")
        print("=" * 96)
        comparison_rows = []
        for m_info in EMBEDDING_MODELS:
            m_id = m_info["id"]
            run_data = await run_evaluation(
                mode=args.mode,
                dataset_source=dataset_source,
                custom_cases=custom_cases,
                top_k=args.top_k,
                similarity_threshold=args.similarity_threshold,
                embedding_model=m_id,
                persist=not args.no_save,
            )
            s = run_data["summary"]
            comparison_rows.append((m_id, run_data["overall_score"], s["hit_rate"], s["mrr"], s["ndcg"], s["context_recall"], s["avg_retrieval_latency_ms"]))
        print(f" {'EMBEDDING MODEL':<26} {'OVERALL':>9} {'HIT@K':>9} {'MRR':>8} {'NDCG':>8} {'RECALL':>9} {'RET LATENCY':>13}")
        print("-" * 96)
        for m_id, ov, hr, mrr, ndcg, rec, lat in comparison_rows:
            print(f" {m_id:<26} {ov * 100:8.1f}% {hr * 100:8.1f}% {mrr:8.3f} {ndcg:8.3f} {rec * 100:8.1f}% {lat:10.1f} ms")
        print("=" * 96)
        return 0

    run_data = await run_evaluation(
        mode=args.mode,
        dataset_source=dataset_source,
        custom_cases=custom_cases,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        embedding_model=args.embedding_model,
        persist=not args.no_save,
    )
    _print_cli_report(run_data)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(run_data, indent=2), encoding="utf-8")
        print(f"Saved JSON report to: {args.output_json}")

    if args.output_md:
        md_report = format_eval_markdown_report(run_data)
        Path(args.output_md).write_text(md_report, encoding="utf-8")
        print(f"Saved Markdown report to: {args.output_md}")

    if args.fail_under is not None:
        pct = run_data.get("overall_score", 0.0) * 100.0
        if pct < args.fail_under:
            print(
                f"ERROR: Overall evaluation score ({pct:.1f}%) is below --fail-under threshold ({args.fail_under:.1f}%).",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Company RAG Chatbot (Retrieval, ReAct Tool Routing, Faithfulness & Relevance)."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "retrieval", "llm_judge"],
        default="full",
        help="Evaluation mode: 'full' (Retrieval + ReAct + Grounded Answer), 'retrieval' (Retrieval-only), or 'llm_judge' (Live LLM + Judge).",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["golden", "synthetic", "combined"],
        default="golden",
        help="Built-in dataset source to evaluate against.",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Path to a custom JSON or JSONL evaluation dataset file.",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Include synthetically generated Q&A test cases from indexed documents in LanceDB.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override CrispEmbed model ID (e.g., all-MiniLM-L6-v2, gte-small, arctic-embed-xs, multilingual-e5-small).",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Benchmark and compare all CrispEmbed models side-by-side.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override Top-K chunks retrieved.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Override similarity threshold (0.0 - 1.0).",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Write full evaluation results to a JSON file.")
    parser.add_argument("--output-md", type=str, default=None, help="Write formatted evaluation report to a Markdown file.")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with non-zero status code if overall percentage score (0-100) is below this threshold.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not persist evaluation run to SQLite.")

    args = parser.parse_args()
    exit_code = asyncio.run(_cli_main_async(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
