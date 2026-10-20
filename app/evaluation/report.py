"""Human-readable evaluation reports (Markdown and terminal)."""

from typing import Any, Dict


def format_eval_markdown_report(run_data: Dict[str, Any]) -> str:
    """Formats an evaluation run dictionary into a GitHub-flavored Markdown report."""
    summary = run_data.get("summary") or {}
    config = run_data.get("config") or {}
    lines = [
        "# RAG & ReAct Agent Evaluation Report",
        "",
        f"- **Dataset:** `{run_data.get('dataset_name')}`",
        f"- **Mode:** `{run_data.get('mode')}`",
        f"- **Embedding Model:** `{run_data.get('embedding_model')}` (`top_k={config.get('top_k')}`, `sim_threshold={config.get('similarity_threshold')}`)",
        f"- **Overall Score:** **{summary.get('overall_score', 0.0) * 100:.1f}%** ({run_data.get('passed_cases')}/{run_data.get('total_cases')} cases passed)",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Score |",
        "| :--- | :---: |",
        f"| **Retrieval Hit Rate @ K** | `{summary.get('hit_rate', 0.0) * 100:.1f}%` |",
        f"| **Mean Reciprocal Rank (MRR)** | `{summary.get('mrr', 0.0):.3f}` |",
        f"| **NDCG @ K** | `{summary.get('ndcg', 0.0):.3f}` |",
        f"| **Context Precision** | `{summary.get('context_precision', 0.0) * 100:.1f}%` |",
        f"| **Context Recall (Fact Coverage)** | `{summary.get('context_recall', 0.0) * 100:.1f}%` |",
        f"| **ReAct Tool Routing Accuracy** | `{summary.get('tool_routing_accuracy', 0.0) * 100:.1f}%` |",
        f"| **Answer Faithfulness** | `{summary.get('faithfulness', 0.0) * 100:.1f}%` |",
        f"| **Answer Relevance** | `{summary.get('answer_relevance', 0.0) * 100:.1f}%` |",
        f"| **Citation Accuracy** | `{summary.get('citation_accuracy', 0.0) * 100:.1f}%` |",
        f"| **RAG Token Budget Compliance** | `{summary.get('budget_compliance_rate', 0.0) * 100:.1f}%` |",
        f"| **Avg Retrieval Latency** | `{summary.get('avg_retrieval_latency_ms', 0.0):.1f} ms` |",
        f"| **Avg End-to-End Latency** | `{summary.get('avg_total_latency_ms', 0.0):.1f} ms` |",
        "",
        "## Per-Case Breakdown",
        "",
        "| ID | Category | Status | Score | Hit@K | MRR | Recall | Routing | Faithfulness | Relevance | Latency |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for result in run_data.get("results", []):
        retrieval = result["retrieval"]
        generation = result["generation"]
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"| `{result['case_id']}` | `{result['category']}` | **{status}** | `{result['case_score'] * 100:.1f}%` | "
            f"`{retrieval['hit_rate']:.2f}` | `{retrieval['mrr']:.2f}` | `{retrieval['context_recall'] * 100:.0f}%` | "
            f"`{generation['tool_routing_score']:.0f}` | `{generation['faithfulness'] * 100:.0f}%` | "
            f"`{generation['answer_relevance'] * 100:.0f}%` | `{result['total_latency_ms']:.1f}ms` |"
        )
    return "\n".join(lines) + "\n"


def print_cli_report(run_data: Dict[str, Any]) -> None:
    """Prints a compact terminal scorecard."""
    summary = run_data.get("summary") or {}
    config = run_data.get("config") or {}
    separator = "=" * 96
    thin = "-" * 96
    print(separator)
    print(
        f" RAG & REACT AGENT EVALUATION SCORECARD // MODE: {run_data.get('mode', 'full').upper()}"
    )
    print(separator)
    print(
        f" Dataset         : {run_data.get('dataset_name')}\n"
        f" Embedding Model : {run_data.get('embedding_model')} (top_k={config.get('top_k')}, sim_threshold={config.get('similarity_threshold')})\n"
        f" Overall Score   : {summary.get('overall_score', 0.0) * 100:.1f}%  [{run_data.get('passed_cases')}/{run_data.get('total_cases')} PASSED]"
    )
    print(thin)
    print(
        f"  Hit Rate @ K   : {summary.get('hit_rate', 0.0) * 100:6.1f}%   |  ReAct Tool Routing : {summary.get('tool_routing_accuracy', 0.0) * 100:6.1f}%\n"
        f"  MRR @ K        : {summary.get('mrr', 0.0):6.3f}    |  Faithfulness       : {summary.get('faithfulness', 0.0) * 100:6.1f}%\n"
        f"  NDCG @ K       : {summary.get('ndcg', 0.0):6.3f}    |  Answer Relevance   : {summary.get('answer_relevance', 0.0) * 100:6.1f}%\n"
        f"  Context Recall : {summary.get('context_recall', 0.0) * 100:6.1f}%   |  Citation Accuracy  : {summary.get('citation_accuracy', 0.0) * 100:6.1f}%\n"
        f"  Ctx Precision  : {summary.get('context_precision', 0.0) * 100:6.1f}%   |  Avg Latency        : {summary.get('avg_total_latency_ms', 0.0):6.1f} ms"
    )
    print(thin)
    print(
        f" {'ID':<10} {'CATEGORY':<14} {'STATUS':<6} {'SCORE':>7} {'HIT':>5} {'MRR':>6} {'RECALL':>7} {'ROUTE':>6} {'FAITH':>7} {'RELEV':>7} {'LATENCY':>9}"
    )
    print(thin)
    for result in run_data.get("results", []):
        retrieval = result["retrieval"]
        generation = result["generation"]
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f" {result['case_id']:<10} {result['category']:<14} {status:<6} {result['case_score'] * 100:6.1f}% "
            f"{retrieval['hit_rate']:5.2f} {retrieval['mrr']:6.2f} {retrieval['context_recall'] * 100:6.0f}% "
            f"{generation['tool_routing_score']:6.1f} {generation['faithfulness'] * 100:6.0f}% "
            f"{generation['answer_relevance'] * 100:6.0f}% {result['total_latency_ms']:7.1f}ms"
        )
    print(separator)
