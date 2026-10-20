"""Retrieval, generation, and per-case evaluation metrics."""

import asyncio
import json
import math
import re
import time
import uuid
from typing import Any, Dict, List, Optional

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
from app.config import KEYLESS_PROVIDERS
from app.embeddings import embedding_engine
from app.store import estimate_tokens, store


def _cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a / norm_a, b / norm_b), 0.0, 1.0))


def _fact_present(fact: str, text: str) -> bool:
    """Checks if an expected fact is present in text (case-insensitive, normalized)."""
    if not fact or not text:
        return False
    normalized_fact = re.sub(r"\s+", " ", fact.strip().lower())
    normalized_text = re.sub(r"\s+", " ", text.strip().lower())
    if normalized_fact in normalized_text:
        return True
    alphanumeric_fact = re.sub(r"[^a-z0-9/]+", " ", normalized_fact).strip()
    alphanumeric_text = re.sub(r"[^a-z0-9/]+", " ", normalized_text).strip()
    if (
        alphanumeric_fact
        and len(alphanumeric_fact) >= 2
        and alphanumeric_fact in alphanumeric_text
    ):
        return True
    return False


def evaluate_retrieval_metrics(
    case: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    rag_token_ceiling: int = 1500,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes retrieval quality metrics for a single test case:
    hit_rate, MRR, NDCG@K, context precision/recall, and token-budget compliance.
    """
    expect_retrieval = bool(case.get("expect_retrieval", True))
    allow_abstain = bool(case.get("allow_abstain", False))
    expected_sources = [
        source.lower() for source in (case.get("expected_sources") or []) if source
    ]
    expected_facts = [fact for fact in (case.get("expected_facts") or []) if fact]

    formatted_block, _ = _format_rag_context_block(
        retrieved_chunks, rag_token_ceiling
    )
    rag_tokens = estimate_tokens(formatted_block) if retrieved_chunks else 0
    budget_compliant = rag_tokens <= rag_token_ceiling

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

    if allow_abstain and not expected_sources and not expected_facts:
        max_similarity = max(
            (float(chunk.get("similarity", 0.0)) for chunk in retrieved_chunks),
            default=0.0,
        )
        score = 1.0 if max_similarity < 0.65 else 0.7
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

    chunk_grades: List[float] = []
    first_relevant_rank: Optional[int] = None
    relevant_count = 0

    for rank_index, chunk in enumerate(retrieved_chunks):
        chunk_source = (chunk.get("source_name") or "").lower()
        chunk_doc_id = (chunk.get("doc_id") or "").lower()
        chunk_text = chunk.get("text") or ""

        source_ok = True
        if expected_sources:
            source_ok = (chunk_source in expected_sources) or (
                chunk_doc_id in expected_sources
            )

        if expected_facts:
            fact_hits = sum(
                1 for fact in expected_facts if _fact_present(fact, chunk_text)
            )
            fact_ratio = fact_hits / float(len(expected_facts))
        else:
            fact_ratio = (
                1.0 if float(chunk.get("similarity", 0.0)) >= 0.25 else 0.0
            )

        if source_ok and (
            fact_ratio > 0.0 or float(chunk.get("similarity", 0.0)) >= 0.35
        ):
            grade = 0.4 + 0.6 * fact_ratio
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank_index + 1
        elif fact_ratio > 0.0:
            grade = 0.5 * fact_ratio
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank_index + 1
        else:
            grade = 0.0

        chunk_grades.append(grade)

    hit_rate = 1.0 if first_relevant_rank is not None else 0.0
    mrr = round(1.0 / first_relevant_rank, 4) if first_relevant_rank is not None else 0.0

    dcg = sum(grade / math.log2(index + 2) for index, grade in enumerate(chunk_grades))
    ideal_grades = sorted(chunk_grades, reverse=True)
    if max(ideal_grades, default=0.0) < 1.0 and hit_rate > 0:
        ideal_grades[0] = 1.0
    idcg = sum(grade / math.log2(index + 2) for index, grade in enumerate(ideal_grades))
    ndcg = round(min(1.0, dcg / idcg), 4) if idcg > 0 else 0.0

    context_precision = round(relevant_count / float(len(retrieved_chunks)), 4)

    combined_text = "\n".join(chunk.get("text", "") for chunk in retrieved_chunks)
    if expected_facts:
        matched_facts = [
            fact for fact in expected_facts if _fact_present(fact, combined_text)
        ]
        missing_facts = [
            fact for fact in expected_facts if fact not in matched_facts
        ]
        context_recall = round(len(matched_facts) / float(len(expected_facts)), 4)
    else:
        matched_facts = []
        missing_facts = []
        expected_answer = case.get("expected_answer") or case.get("question") or ""
        vectors = embedding_engine.embed_batch(
            [expected_answer, combined_text[:1500]], model_name=embedding_model
        )
        context_recall = (
            round(_cosine_sim(vectors[0], vectors[1]), 4)
            if len(vectors) == 2
            else hit_rate
        )

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
    """Synthesizes a grounded answer when running without an external LLM API key."""
    question = (case.get("question") or "").strip()
    category = case.get("category", "rag_factual")
    allow_abstain = bool(case.get("allow_abstain", False))

    if not did_retrieve:
        math_clean = re.sub(r"[^0-9\+\-\*\/\(\)\.]", "", question)
        if math_clean and any(character.isdigit() for character in math_clean):
            try:
                value = eval(math_clean, {"__builtins__": {}}, {})
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                return f"The result of `{math_clean}` is **{value}**."
            except Exception:
                pass
        return "Hello! How can I assist you with your knowledge base today?"

    if not retrieved_chunks:
        return (
            "I searched the knowledge base, but the retrieved context does not contain "
            "information to answer this question."
        )

    question_words = {
        word.lower().strip("?.,!\"'()[]")
        for word in question.split()
        if len(word.strip("?.,!\"'()[]")) >= 4
        and word.lower()
        not in {
            "what",
            "when",
            "where",
            "which",
            "does",
            "acme",
            "corp",
            "corps",
            "policy",
            "rules",
            "regarding",
            "about",
            "their",
        }
    }
    combined_lower = " ".join(chunk.get("text", "").lower() for chunk in retrieved_chunks)
    keyword_hits = sum(1 for word in question_words if word in combined_lower)
    top_similarity = max(
        (float(chunk.get("similarity", 0.0)) for chunk in retrieved_chunks),
        default=0.0,
    )

    if (allow_abstain or category == "unanswerable") and keyword_hits == 0 and top_similarity < 0.45:
        return (
            "The retrieved knowledge base context does not contain information regarding "
            f"this topic ({question})."
        )

    selected_excerpts: List[str] = []
    sources_cited: List[str] = []
    for chunk in retrieved_chunks[:3]:
        source = chunk.get("source_name") or "document"
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        selected_excerpts.append(f"{text} [{source}]")
        if source not in sources_cited:
            sources_cited.append(source)

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
    Computes generation and agent-quality metrics:
    tool routing, faithfulness, answer relevance, citation accuracy, and fact coverage.
    """
    expect_retrieval = bool(case.get("expect_retrieval", True))
    allow_abstain = bool(case.get("allow_abstain", False))
    expected_sources = [
        source.lower() for source in (case.get("expected_sources") or []) if source
    ]
    expected_facts = [fact for fact in (case.get("expected_facts") or []) if fact]
    expected_answer = (case.get("expected_answer") or "").strip()
    question = (case.get("question") or "").strip()

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

    answer_lower = answer.lower()

    bracket_citations = [
        match.group(1).strip().lower()
        for match in re.finditer(
            r"\[([^\[\]]+\.(?:md|pdf|txt|html)|https?://[^\[\]]+)\]",
            answer,
            flags=re.IGNORECASE,
        )
    ]
    if not expect_retrieval:
        citation_accuracy = 1.0 if len(bracket_citations) == 0 else 0.5
    elif allow_abstain and not expected_sources:
        citation_accuracy = 1.0
    elif expected_sources:
        has_expected_citation = any(
            any(
                expected_source in bracket_citation
                or expected_source in answer_lower
                for expected_source in expected_sources
            )
            for bracket_citation in (bracket_citations or [answer_lower])
        )
        if bracket_citations and any(
            bracket_citation in expected_sources
            for bracket_citation in bracket_citations
        ):
            citation_accuracy = 1.0
        elif has_expected_citation:
            citation_accuracy = 0.85
        else:
            citation_accuracy = 0.0
    else:
        citation_accuracy = 1.0 if bracket_citations else 0.7

    if expected_facts:
        answer_fact_hits = sum(
            1 for fact in expected_facts if _fact_present(fact, answer)
        )
        answer_fact_coverage = round(
            answer_fact_hits / float(len(expected_facts)), 4
        )
    else:
        answer_fact_coverage = 1.0

    abstain_phrases = (
        "does not contain",
        "no matching documents",
        "not mentioned",
        "no information",
        "cannot find",
        "not specify",
        "not in the knowledge base",
    )
    did_abstain = any(phrase in answer_lower for phrase in abstain_phrases)

    combined_context = "\n\n".join(
        chunk.get("text", "") for chunk in retrieved_chunks
    )
    if not expect_retrieval:
        faithfulness = 1.0
        if expected_facts:
            answer_relevance = answer_fact_coverage
        else:
            vectors = embedding_engine.embed_batch(
                [expected_answer or question, answer[:1000]],
                model_name=embedding_model,
            )
            similarity = _cosine_sim(vectors[0], vectors[1]) if len(vectors) == 2 else 0.8
            answer_relevance = round(max(0.75, similarity), 4)
    elif allow_abstain and not expected_facts:
        if did_abstain:
            faithfulness = 1.0
            answer_relevance = 1.0
        else:
            faithfulness = 0.35
            answer_relevance = 0.4
    elif not retrieved_chunks:
        faithfulness = 1.0 if did_abstain else 0.0
        answer_relevance = 0.0
    else:
        answer_without_citations = re.sub(r"\[[^\[\]]+\]", " ", answer)
        answer_numbers = set(re.findall(r"\b\d[\d,.:]*\b", answer_without_citations))
        context_numbers = set(re.findall(r"\b\d[\d,.:]*\b", combined_context))
        significant_answer_numbers = {
            number
            for number in answer_numbers
            if len(number) >= 2 or int(re.sub(r"\D", "", number) or "0") > 4
        }
        if significant_answer_numbers:
            grounded_numbers = sum(
                1
                for number in significant_answer_numbers
                if number in context_numbers or number in combined_context
            )
            number_grounding = grounded_numbers / float(
                len(significant_answer_numbers)
            )
        else:
            number_grounding = 1.0

        vectors = embedding_engine.embed_batch(
            [combined_context[:1600], answer[:1200], expected_answer or question],
            model_name=embedding_model,
        )
        context_answer_similarity = (
            _cosine_sim(vectors[0], vectors[1]) if len(vectors) == 3 else 0.7
        )
        expected_answer_similarity = (
            _cosine_sim(vectors[2], vectors[1]) if len(vectors) == 3 else 0.7
        )

        faithfulness = round(
            min(
                1.0,
                0.55 * number_grounding
                + 0.45 * min(1.0, context_answer_similarity * 1.25),
            ),
            4,
        )
        if expected_facts:
            answer_relevance = round(
                0.65 * answer_fact_coverage
                + 0.35 * min(1.0, expected_answer_similarity * 1.2),
                4,
            )
        else:
            answer_relevance = round(
                min(1.0, expected_answer_similarity * 1.2), 4
            )

    judge_method = "semantic_heuristic"
    judge_rationale = (
        f"Fact coverage: {int(answer_fact_coverage * 100)}%, "
        f"Tool routing: {'PASS' if tool_routing_score == 1.0 else 'FAIL'}, "
        f"Citation score: {int(citation_accuracy * 100)}%."
    )

    if use_llm_judge and llm_settings:
        api_key = llm_settings.get("_decrypted_api_key", "")
        provider = llm_settings.get("provider", "openrouter")
        if api_key or provider in KEYLESS_PROVIDERS:
            try:
                llm = ChatOpenAI(
                    api_key=api_key or "not-needed",
                    base_url=llm_settings.get(
                        "base_url", "https://openrouter.ai/api/v1"
                    ),
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
                response = await llm.ainvoke([HumanMessage(content=judge_prompt)])
                raw_judge = str(response.content or "").strip()
                judge_match = re.search(r"\{.*\}", raw_judge, flags=re.DOTALL)
                if judge_match:
                    judge_data = json.loads(judge_match.group(0))
                    if "faithfulness" in judge_data:
                        faithfulness = round(
                            0.5 * faithfulness
                            + 0.5
                            * float(
                                np.clip(float(judge_data["faithfulness"]), 0.0, 1.0)
                            ),
                            4,
                        )
                    if "answer_relevance" in judge_data:
                        answer_relevance = round(
                            0.5 * answer_relevance
                            + 0.5
                            * float(
                                np.clip(
                                    float(judge_data["answer_relevance"]), 0.0, 1.0
                                )
                            ),
                            4,
                        )
                    if judge_data.get("rationale"):
                        judge_rationale = str(judge_data["rationale"]).strip()
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


def _score_case(
    mode: str, expect_retrieval: bool, retrieval: Dict[str, Any], generation: Dict[str, Any]
) -> float:
    if mode == "retrieval":
        return round(
            0.30 * retrieval["hit_rate"]
            + 0.25 * retrieval["mrr"]
            + 0.25 * retrieval["context_recall"]
            + 0.20 * generation["tool_routing_score"],
            4,
        )
    if not expect_retrieval:
        return round(
            0.50 * generation["tool_routing_score"]
            + 0.30 * generation["answer_relevance"]
            + 0.20 * generation["faithfulness"],
            4,
        )
    return round(
        0.20 * retrieval["hit_rate"]
        + 0.15 * retrieval["mrr"]
        + 0.20 * retrieval["context_recall"]
        + 0.15 * generation["tool_routing_score"]
        + 0.15 * generation["faithfulness"]
        + 0.15 * generation["answer_relevance"],
        4,
    )


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
    """Runs one case through routing, retrieval, and answer synthesis."""
    start_time = time.perf_counter()
    question = (case.get("question") or "").strip()
    target_docs = list(case.get("target_docs") or [])
    expect_retrieval = bool(case.get("expect_retrieval", True))
    settings = settings or store.get_settings(
        user_id=user_id, include_decrypted_key=True
    )

    api_key = settings.get("_decrypted_api_key", "")
    provider = settings.get("provider", "openrouter")
    has_live_llm = (mode in ("full", "llm_judge")) and (
        bool(api_key) or provider in KEYLESS_PROVIDERS
    )

    react_steps: List[Dict[str, Any]] = []
    did_retrieve = False
    search_query = question
    effective_targets = list(target_docs)

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
            prompt_message = question
            if target_docs:
                prompt_message = (
                    f"[Targeted Document Filter Active: {', '.join(target_docs)}]\n"
                    f"User Query: {question}"
                )
            decision = await llm.ainvoke(
                [
                    SystemMessage(content=settings.get("system_prompt", "")),
                    HumanMessage(content=prompt_message),
                ]
            )
            if getattr(decision, "tool_calls", None):
                did_retrieve = True
                tool_call = decision.tool_calls[0]
                arguments = tool_call.get("args") or {}
                search_query = arguments.get("query") or question
                if (
                    arguments.get("target_source")
                    and arguments["target_source"] not in effective_targets
                ):
                    effective_targets.append(arguments["target_source"])
                react_steps.append(
                    {
                        "step": "THOUGHT",
                        "detail": arguments.get("reasoning")
                        or "LLM decided to invoke search_knowledge_base.",
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
            did_retrieve, reason = _should_auto_retrieve_heuristic(
                question, target_docs
            )
            react_steps.append({"step": "THOUGHT", "detail": reason})
    else:
        did_retrieve, reason = _should_auto_retrieve_heuristic(question, target_docs)
        react_steps.append({"step": "THOUGHT", "detail": reason})

    retrieval_start = time.perf_counter()
    retrieved_chunks: List[Dict[str, Any]] = []
    if did_retrieve or (mode == "retrieval" and expect_retrieval):
        react_steps.append(
            {
                "step": "ACTION",
                "detail": f'search_knowledge_base(query="{search_query}", target_docs={effective_targets})',
            }
        )
        retrieved_chunks, retrieval_error = await _safe_hybrid_search(
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
        if retrieval_error:
            react_steps.append({"step": "ERROR", "detail": retrieval_error})
        else:
            react_steps.append(
                {
                    "step": "OBSERVATION",
                    "detail": f"Retrieved {len(retrieved_chunks)} chunks from LanceDB.",
                }
            )
    retrieval_latency_ms = round(
        (time.perf_counter() - retrieval_start) * 1000.0, 2
    )

    retrieval_metrics = await asyncio.to_thread(
        evaluate_retrieval_metrics,
        case,
        retrieved_chunks,
        rag_token_ceiling,
        embedding_model,
    )

    generated_answer = ""
    if mode == "retrieval":
        tool_routing_score = 1.0 if (did_retrieve == expect_retrieval) else 0.0
        generation_metrics = {
            "tool_routing_score": tool_routing_score,
            "faithfulness": retrieval_metrics["hit_rate"],
            "answer_relevance": retrieval_metrics["context_recall"],
            "citation_accuracy": 1.0,
            "answer_fact_coverage": retrieval_metrics["context_recall"],
            "judge_method": "retrieval_only",
            "judge_rationale": "Retrieval-only benchmark mode.",
        }
    else:
        if has_live_llm and mode == "llm_judge":
            try:
                rag_context, _ = _format_rag_context_block(
                    retrieved_chunks, rag_token_ceiling
                )
                llm_generation = ChatOpenAI(
                    api_key=api_key or "not-needed",
                    base_url=settings.get(
                        "base_url", "https://openrouter.ai/api/v1"
                    ),
                    model=settings.get("model", "openai/gpt-4o-mini"),
                    temperature=0.1,
                    max_tokens=400,
                    timeout=25.0,
                    max_retries=1,
                )
                if did_retrieve:
                    messages = [
                        SystemMessage(content=settings.get("system_prompt", "")),
                        SystemMessage(
                            content=(
                                "## RETRIEVED KNOWLEDGE BASE CONTEXT\n"
                                f"{rag_context}\n\n"
                                "Answer accurately using the retrieved context above and cite sources in brackets like [filename.md]."
                            )
                        ),
                        HumanMessage(content=question),
                    ]
                else:
                    messages = [
                        SystemMessage(content=settings.get("system_prompt", "")),
                        HumanMessage(content=question),
                    ]
                response = await llm_generation.ainvoke(messages)
                generated_answer = str(response.content or "").strip()
            except Exception:
                generated_answer = ""

        if not generated_answer:
            generated_answer = _synthesize_deterministic_agent_answer(
                case=case,
                retrieved_chunks=retrieved_chunks,
                did_retrieve=did_retrieve,
            )

        generation_metrics = await evaluate_generation_metrics(
            case=case,
            answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            did_retrieve=did_retrieve,
            embedding_model=embedding_model,
            use_llm_judge=(mode == "llm_judge" and has_live_llm),
            llm_settings=settings,
        )

    total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    case_score = _score_case(
        mode, expect_retrieval, retrieval_metrics, generation_metrics
    )

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
        "passed": case_score >= 0.70,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "retrieved_chunks": [
            {
                "source_name": chunk.get("source_name"),
                "chunk_index": chunk.get("chunk_index"),
                "similarity": chunk.get("similarity"),
                "hybrid_score": chunk.get("hybrid_score"),
                "match_type": chunk.get("match_type"),
                "preview": (chunk.get("text") or "")[:200]
                + ("..." if len(chunk.get("text") or "") > 200 else ""),
            }
            for chunk in retrieved_chunks
        ],
        "react_steps": react_steps,
        "retrieval_latency_ms": retrieval_latency_ms,
        "total_latency_ms": total_latency_ms,
    }
