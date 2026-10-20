"""Evaluation suite, re-exported for backwards-compatible `app.evaluation` imports."""

from app.evaluation.cli import main
from app.evaluation.datasets import (
    GOLDEN_EVAL_DATASET,
    SAMPLE_COMPANY_HANDBOOK_TEXT,
    ensure_default_handbook_seeded,
)
from app.evaluation.metrics import (
    _cosine_sim,
    _fact_present,
    _synthesize_deterministic_agent_answer,
    evaluate_generation_metrics,
    evaluate_retrieval_metrics,
    evaluate_single_case,
)
from app.evaluation.report import format_eval_markdown_report, print_cli_report
from app.evaluation.runner import (
    _aggregate_eval_results,
    resolve_eval_dataset,
    run_evaluation,
    run_evaluation_stream,
)
from app.evaluation.synthetic import (
    _extract_facts_from_chunk,
    generate_synthetic_eval_cases,
)

__all__ = [
    "GOLDEN_EVAL_DATASET",
    "SAMPLE_COMPANY_HANDBOOK_TEXT",
    "_aggregate_eval_results",
    "_cosine_sim",
    "_extract_facts_from_chunk",
    "_fact_present",
    "_synthesize_deterministic_agent_answer",
    "ensure_default_handbook_seeded",
    "evaluate_generation_metrics",
    "evaluate_retrieval_metrics",
    "evaluate_single_case",
    "format_eval_markdown_report",
    "generate_synthetic_eval_cases",
    "main",
    "print_cli_report",
    "resolve_eval_dataset",
    "run_evaluation",
    "run_evaluation_stream",
]
