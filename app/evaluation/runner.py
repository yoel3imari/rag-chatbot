"""Evaluation dataset resolution, aggregation, and execution."""

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.evaluation.datasets import (
    GOLDEN_EVAL_DATASET,
    ensure_default_handbook_seeded,
)
from app.evaluation.metrics import evaluate_single_case
from app.evaluation.synthetic import generate_synthetic_eval_cases
from app.models import EvalCaseModel
from app.store import store
from app.vector_store import vector_store


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

    passed_count = sum(1 for result in results if result["passed"])
    retrieval_cases = [
        result for result in results if result.get("expect_retrieval", True)
    ]
    retrieval_denominator = len(retrieval_cases) or total

    def average_retrieval(metric: str) -> float:
        return round(
            sum(
                result["retrieval"][metric]
                for result in (retrieval_cases or results)
            )
            / float(retrieval_denominator),
            4,
        )

    def average_generation(metric: str) -> float:
        return round(
            sum(result["generation"][metric] for result in results)
            / float(total),
            4,
        )

    avg_hit_rate = average_retrieval("hit_rate")
    avg_mrr = average_retrieval("mrr")
    avg_ndcg = average_retrieval("ndcg")
    avg_context_precision = average_retrieval("context_precision")
    avg_context_recall = average_retrieval("context_recall")

    avg_tool_routing = average_generation("tool_routing_score")
    avg_faithfulness = average_generation("faithfulness")
    avg_answer_relevance = average_generation("answer_relevance")
    avg_citation_accuracy = average_generation("citation_accuracy")

    budget_compliance_rate = round(
        sum(
            1.0 if result["retrieval"]["budget_compliant"] else 0.0
            for result in results
        )
        / float(total),
        4,
    )
    avg_retrieval_latency = round(
        sum(result["retrieval_latency_ms"] for result in results)
        / float(total),
        2,
    )
    avg_total_latency = round(
        sum(result["total_latency_ms"] for result in results) / float(total), 2
    )
    overall_score = round(
        sum(result["case_score"] for result in results) / float(total), 4
    )

    categories: Dict[str, Dict[str, Any]] = {}
    for result in results:
        category = result["category"]
        if category not in categories:
            categories[category] = {"count": 0, "passed": 0, "score_sum": 0.0}
        categories[category]["count"] += 1
        if result["passed"]:
            categories[category]["passed"] += 1
        categories[category]["score_sum"] += result["case_score"]

    category_breakdown = {
        category: {
            "count": info["count"],
            "passed": info["passed"],
            "avg_score": round(info["score_sum"] / float(info["count"]), 4),
        }
        for category, info in categories.items()
    }

    summary = {
        "overall_score": overall_score,
        "pass_rate": round(passed_count / float(total), 4),
        "hit_rate": avg_hit_rate,
        "mrr": avg_mrr,
        "ndcg": avg_ndcg,
        "context_precision": avg_context_precision,
        "context_recall": avg_context_recall,
        "tool_routing_accuracy": avg_tool_routing,
        "faithfulness": avg_faithfulness,
        "answer_relevance": avg_answer_relevance,
        "citation_accuracy": avg_citation_accuracy,
        "budget_compliance_rate": budget_compliance_rate,
        "avg_retrieval_latency_ms": avg_retrieval_latency,
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
    """Resolves the requested dataset ('golden', 'synthetic', 'combined', or 'custom')."""
    ensure_default_handbook_seeded()

    if dataset_source == "custom" and custom_cases:
        normalized = [
            case.model_dump() if isinstance(case, EvalCaseModel) else dict(case)
            for case in custom_cases
        ]
        return f"Custom Dataset ({len(normalized)} cases)", normalized

    if dataset_source == "synthetic":
        synthetic_cases = await generate_synthetic_eval_cases(
            max_cases=max_synthetic_cases, user_id=user_id
        )
        return (
            f"Synthetic Corpus Suite ({len(synthetic_cases)} cases)",
            synthetic_cases,
        )

    if dataset_source == "combined":
        synthetic_cases = await generate_synthetic_eval_cases(
            max_cases=max_synthetic_cases, user_id=user_id
        )
        combined = list(GOLDEN_EVAL_DATASET) + synthetic_cases
        return f"Golden + Synthetic Suite ({len(combined)} cases)", combined

    return (
        f"Golden Benchmark Suite ({len(GOLDEN_EVAL_DATASET)} cases)",
        list(GOLDEN_EVAL_DATASET),
    )


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
    """Runs a complete evaluation suite and returns the persisted scorecard."""
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
    Async generator yielding real-time evaluation progress:
    eval_start -> case_result* -> eval_complete.
    """
    settings = store.get_settings(user_id=user_id, include_decrypted_key=True)
    effective_top_k = int(
        top_k if top_k is not None else settings.get("top_k", 5)
    )
    effective_similarity = float(
        similarity_threshold
        if similarity_threshold is not None
        else settings.get("similarity_threshold", 0.15)
    )
    effective_dedup = float(
        dedup_threshold
        if dedup_threshold is not None
        else settings.get("dedup_threshold", 0.92)
    )
    effective_embedding_model = str(
        embedding_model or settings.get("embedding_model", "all-MiniLM-L6-v2")
    )
    rag_ceiling = int(settings.get("rag_token_ceiling", 1500))
    llm_model = (
        f"{settings.get('provider', 'openrouter')}/"
        f"{settings.get('model', 'openai/gpt-4o-mini')}"
    )

    await asyncio.to_thread(
        vector_store.reindex_for_model, effective_embedding_model
    )

    dataset_name, cases = await resolve_eval_dataset(
        dataset_source=dataset_source,
        custom_cases=custom_cases,
        max_synthetic_cases=max_synthetic_cases,
        user_id=user_id,
    )

    config = {
        "mode": mode,
        "dataset_source": dataset_source,
        "top_k": effective_top_k,
        "similarity_threshold": effective_similarity,
        "dedup_threshold": effective_dedup,
        "embedding_model": effective_embedding_model,
        "rag_token_ceiling": rag_ceiling,
    }

    yield {
        "type": "eval_start",
        "dataset_name": dataset_name,
        "total_cases": len(cases),
        "config": config,
    }

    results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_result = await evaluate_single_case(
            case=case,
            mode=mode,
            top_k=effective_top_k,
            similarity_threshold=effective_similarity,
            dedup_threshold=effective_dedup,
            rag_token_ceiling=rag_ceiling,
            embedding_model=effective_embedding_model,
            settings=settings,
            user_id=user_id,
        )
        results.append(case_result)
        progress = round(((index + 1) / float(max(1, len(cases)))) * 100)
        yield {
            "type": "case_result",
            "index": index + 1,
            "total_cases": len(cases),
            "progress": progress,
            "result": case_result,
        }

    run_record = _aggregate_eval_results(
        results=results,
        mode=mode,
        dataset_name=dataset_name,
        embedding_model=effective_embedding_model,
        llm_model=llm_model,
        config=config,
    )

    if persist:
        saved = store.save_eval_run(run_record, user_id=user_id)
        if saved:
            run_record = saved

    yield {"type": "eval_complete", "progress": 100, "run": run_record}
