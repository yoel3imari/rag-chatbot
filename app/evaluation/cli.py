"""Command-line entry point for the offline evaluation suite."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.config import EMBEDDING_MODELS
from app.evaluation.report import format_eval_markdown_report, print_cli_report
from app.evaluation.runner import run_evaluation


def _load_custom_dataset_file(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset file not found: {path_str}"
        )
    raw = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    if isinstance(data, dict) and "cases" in data:
        return list(data["cases"])
    if isinstance(data, list):
        return data
    raise ValueError(
        "Dataset file must contain a JSON array of test cases or an object with a 'cases' key."
    )


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
        for model_info in EMBEDDING_MODELS:
            model_id = model_info["id"]
            run_data = await run_evaluation(
                mode=args.mode,
                dataset_source=dataset_source,
                custom_cases=custom_cases,
                top_k=args.top_k,
                similarity_threshold=args.similarity_threshold,
                embedding_model=model_id,
                persist=not args.no_save,
            )
            summary = run_data["summary"]
            comparison_rows.append(
                (
                    model_id,
                    run_data["overall_score"],
                    summary["hit_rate"],
                    summary["mrr"],
                    summary["ndcg"],
                    summary["context_recall"],
                    summary["avg_retrieval_latency_ms"],
                )
            )
        print(
            f" {'EMBEDDING MODEL':<26} {'OVERALL':>9} {'HIT@K':>9} {'MRR':>8} {'NDCG':>8} {'RECALL':>9} {'RET LATENCY':>13}"
        )
        print("-" * 96)
        for (
            model_id,
            overall,
            hit_rate,
            mrr,
            ndcg,
            recall,
            latency,
        ) in comparison_rows:
            print(
                f" {model_id:<26} {overall * 100:8.1f}% {hit_rate * 100:8.1f}% {mrr:8.3f} {ndcg:8.3f} {recall * 100:8.1f}% {latency:10.1f} ms"
            )
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
    print_cli_report(run_data)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(run_data, indent=2), encoding="utf-8"
        )
        print(f"Saved JSON report to: {args.output_json}")

    if args.output_md:
        markdown_report = format_eval_markdown_report(run_data)
        Path(args.output_md).write_text(markdown_report, encoding="utf-8")
        print(f"Saved Markdown report to: {args.output_md}")

    if args.fail_under is not None:
        percentage = run_data.get("overall_score", 0.0) * 100.0
        if percentage < args.fail_under:
            print(
                f"ERROR: Overall evaluation score ({percentage:.1f}%) is below --fail-under threshold ({args.fail_under:.1f}%).",
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
    parser.add_argument(
        "--top-k", type=int, default=None, help="Override Top-K chunks retrieved."
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Override similarity threshold (0.0 - 1.0).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write full evaluation results to a JSON file.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write formatted evaluation report to a Markdown file.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with non-zero status code if overall percentage score (0-100) is below this threshold.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist evaluation run to SQLite.",
    )

    args = parser.parse_args()
    exit_code = asyncio.run(_cli_main_async(args))
    raise SystemExit(exit_code)
