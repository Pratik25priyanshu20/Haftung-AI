"""Generate LaTeX and Markdown tables for research paper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(val: Any, decimals: int = 3) -> str:
    """Format a value for table display."""
    if val is None:
        return "---"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _bold_best(values: dict[str, float], lower_is_better: bool = False) -> dict[str, str]:
    """Return formatted strings with best value bolded."""
    if not values:
        return {}
    best_val = min(values.values()) if lower_is_better else max(values.values())
    return {
        k: f"**{_fmt(v)}**" if v == best_val else _fmt(v)
        for k, v in values.items()
    }


def _bold_best_latex(values: dict[str, float], lower_is_better: bool = False) -> dict[str, str]:
    """Return LaTeX-formatted strings with best value bolded."""
    if not values:
        return {}
    best_val = min(values.values()) if lower_is_better else max(values.values())
    return {
        k: f"\\textbf{{{_fmt(v)}}}" if v == best_val else _fmt(v)
        for k, v in values.items()
    }


# Metrics where lower values are better
LOWER_IS_BETTER = {
    "responsibility_mae", "ece", "brier_score", "hallucination_rate",
    "avg_entropy", "nli_hallucination_rate", "nli_contradiction_rate",
    "mean_latency_s", "mean_retrieval_latency_s",
}

# Main metrics for Table 1
TABLE1_METRICS = [
    "causation_accuracy_taxonomy",
    "factors_f1",
    "responsibility_mae",
    "nli_hallucination_rate",
    "nli_contradiction_rate",
    "ece",
    "brier_score",
    "mean_latency_s",
]

TABLE1_LABELS = {
    "causation_accuracy_taxonomy": "Causation Accuracy (Taxonomy)",
    "factors_f1": "Contributing Factors F1",
    "responsibility_mae": "Responsibility MAE",
    "nli_hallucination_rate": "NLI Hallucination Rate",
    "nli_contradiction_rate": "NLI Contradiction Rate",
    "ece": "ECE",
    "brier_score": "Brier Score",
    "mean_latency_s": "Mean Latency (s)",
}


def generate_main_results_table(
    metrics: dict[str, dict[str, Any]],
    stat_tests: dict[str, Any] | None = None,
    fmt: str = "markdown",
) -> str:
    """Generate Table 1: S1 vs S2 vs S3 main results with optional p-values.

    Args:
        metrics: Dict mapping variant -> metric dict.
        stat_tests: Optional statistical test results for p-values.
        fmt: Output format ("markdown" or "latex").
    """
    variants = sorted(metrics.keys())
    available_metrics = [m for m in TABLE1_METRICS if any(m in metrics[v] for v in variants)]

    if fmt == "latex":
        return _main_table_latex(metrics, variants, available_metrics, stat_tests)
    return _main_table_markdown(metrics, variants, available_metrics, stat_tests)


def _main_table_markdown(
    metrics: dict[str, dict[str, Any]],
    variants: list[str],
    metric_list: list[str],
    stat_tests: dict[str, Any] | None,
) -> str:
    """Markdown format for Table 1."""
    # Header
    cols = ["Metric"] + variants
    if stat_tests and "S1_vs_S2" in stat_tests.get("pairwise", {}):
        cols.append("p (S1 vs S2)")
    if stat_tests and "S1_vs_S3" in stat_tests.get("pairwise", {}):
        cols.append("p (S1 vs S3)")

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = [header, sep]

    for metric_name in metric_list:
        label = TABLE1_LABELS.get(metric_name, metric_name)
        lib = metric_name in LOWER_IS_BETTER

        vals = {v: metrics[v].get(metric_name) for v in variants}
        float_vals = {k: v for k, v in vals.items() if isinstance(v, (int, float))}
        formatted = _bold_best(float_vals, lower_is_better=lib) if float_vals else {}

        row = [label]
        for v in variants:
            row.append(formatted.get(v, _fmt(vals.get(v))))

        # Add p-values if available
        if stat_tests:
            pairwise = stat_tests.get("pairwise", {})
            for pair in ["S1_vs_S2", "S1_vs_S3"]:
                if pair in pairwise:
                    # Find matching metric in stat tests
                    pair_data = pairwise[pair]
                    # Try taxonomy_accuracy for causation_accuracy_taxonomy
                    test_metric = metric_name
                    if test_metric == "causation_accuracy_taxonomy":
                        test_metric = "taxonomy_accuracy"
                    if test_metric in pair_data:
                        p = pair_data[test_metric]["paired_t_test"]["p_value"]
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        row.append(f"{p:.4f}{sig}")
                    else:
                        row.append("---")

        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def _main_table_latex(
    metrics: dict[str, dict[str, Any]],
    variants: list[str],
    metric_list: list[str],
    stat_tests: dict[str, Any] | None,
) -> str:
    """LaTeX format for Table 1."""
    n_cols = len(variants) + 1
    col_spec = "l" + "c" * len(variants)
    if stat_tests:
        col_spec += "c"
        n_cols += 1

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Main results: S1 (No-RAG) vs S2 (RAG) vs S3 (RAG+Constraints)}",
        "\\label{tab:main-results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header
    header_parts = ["Metric"] + variants
    if stat_tests:
        header_parts.append("$p$ (S1 vs S2)")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for metric_name in metric_list:
        label = TABLE1_LABELS.get(metric_name, metric_name)
        lib = metric_name in LOWER_IS_BETTER

        vals = {v: metrics[v].get(metric_name) for v in variants}
        float_vals = {k: v for k, v in vals.items() if isinstance(v, (int, float))}
        formatted = _bold_best_latex(float_vals, lower_is_better=lib) if float_vals else {}

        row_parts = [label]
        for v in variants:
            row_parts.append(formatted.get(v, _fmt(vals.get(v))))

        if stat_tests:
            pairwise = stat_tests.get("pairwise", {})
            pair_data = pairwise.get("S1_vs_S2", {})
            test_metric = "taxonomy_accuracy" if metric_name == "causation_accuracy_taxonomy" else metric_name
            if test_metric in pair_data:
                p = pair_data[test_metric]["paired_t_test"]["p_value"]
                row_parts.append(f"{p:.4f}")
            else:
                row_parts.append("---")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_per_category_table(
    per_category: dict[str, dict[str, dict[str, float]]],
    fmt: str = "markdown",
) -> str:
    """Generate Table 2: per-category accuracy breakdown."""
    categories = sorted(per_category.keys())
    variants = sorted(
        set(v for cat_data in per_category.values() for v in cat_data.keys())
    )

    if fmt == "latex":
        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{Per-category causation accuracy}",
            "\\label{tab:per-category}",
            "\\begin{tabular}{l" + "c" * len(variants) + "}",
            "\\toprule",
            " & ".join(["Category"] + variants) + " \\\\",
            "\\midrule",
        ]
        for cat in categories:
            vals = {v: per_category[cat].get(v, {}).get("accuracy", 0.0) for v in variants}
            formatted = _bold_best_latex(vals)
            row = [cat.replace("_", " ").title()] + [formatted.get(v, "---") for v in variants]
            lines.append(" & ".join(row) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        return "\n".join(lines)

    # Markdown
    header = "| Category | " + " | ".join(variants) + " |"
    sep = "|---|" + "|".join(["---"] * len(variants)) + "|"
    rows = [header, sep]
    for cat in categories:
        vals = {v: per_category[cat].get(v, {}).get("accuracy", 0.0) for v in variants}
        formatted = _bold_best(vals)
        row = f"| {cat.replace('_', ' ').title()} | " + " | ".join(formatted.get(v, "---") for v in variants) + " |"
        rows.append(row)
    return "\n".join(rows)


def generate_retrieval_quality_table(
    metrics: dict[str, dict[str, Any]],
    fmt: str = "markdown",
) -> str:
    """Generate Table 3: retrieval quality metrics for S2/S3."""
    retrieval_metrics = ["precision_at_5", "mrr", "ndcg_at_5", "mean_retrieval_latency_s"]
    retrieval_labels = {
        "precision_at_5": "Precision@5",
        "mrr": "MRR",
        "ndcg_at_5": "nDCG@5",
        "mean_retrieval_latency_s": "Retrieval Latency (s)",
    }

    variants = [v for v in sorted(metrics.keys()) if v in ("S2", "S3")]
    if not variants:
        return "No retrieval variants found."

    available = [m for m in retrieval_metrics if any(m in metrics.get(v, {}) for v in variants)]

    if fmt == "latex":
        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{Retrieval quality metrics (S2 and S3 only)}",
            "\\label{tab:retrieval}",
            "\\begin{tabular}{l" + "c" * len(variants) + "}",
            "\\toprule",
            " & ".join(["Metric"] + variants) + " \\\\",
            "\\midrule",
        ]
        for m in available:
            label = retrieval_labels.get(m, m)
            row = [label]
            for v in variants:
                val = metrics.get(v, {}).get(m)
                row.append(_fmt(val))
            lines.append(" & ".join(row) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        return "\n".join(lines)

    # Markdown
    header = "| Metric | " + " | ".join(variants) + " |"
    sep = "|---|" + "|".join(["---"] * len(variants)) + "|"
    rows = [header, sep]
    for m in available:
        label = retrieval_labels.get(m, m)
        row = f"| {label} | " + " | ".join(_fmt(metrics.get(v, {}).get(m)) for v in variants) + " |"
        rows.append(row)
    return "\n".join(rows)


def generate_all_tables(
    results_dir: Path,
    fmt: str = "markdown",
    text_mode: bool = False,
) -> str:
    """Generate all research tables from results directory."""
    comp_name = "comparative_text_metrics.json" if text_mode else "comparative_metrics.json"
    comp_path = results_dir / comp_name
    if not comp_path.exists():
        return f"No comparative metrics found at {comp_path}"

    with open(comp_path) as f:
        metrics = json.load(f)

    # Load statistical tests if available
    stat_suffix = "_text" if text_mode else ""
    stat_path = results_dir / f"statistical_tests{stat_suffix}.json"
    stat_tests = None
    if stat_path.exists():
        with open(stat_path) as f:
            stat_tests = json.load(f)

    sections: list[str] = []

    # Table 1: Main results
    sections.append("## Table 1: Main Results\n")
    sections.append(generate_main_results_table(metrics, stat_tests, fmt))

    # Table 2: Per-category (if available)
    cat_path = results_dir / "per_category_comparison.json"
    if cat_path.exists():
        with open(cat_path) as f:
            per_cat = json.load(f)
        sections.append("\n\n## Table 2: Per-Category Breakdown\n")
        sections.append(generate_per_category_table(per_cat, fmt))

    # Table 3: Retrieval quality
    sections.append("\n\n## Table 3: Retrieval Quality (S2/S3)\n")
    sections.append(generate_retrieval_quality_table(metrics, fmt))

    output = "\n".join(sections)

    # Save to file
    ext = "tex" if fmt == "latex" else "md"
    output_path = results_dir / f"results_tables.{ext}"
    with open(output_path, "w") as f:
        f.write(output)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate results tables")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--text", action="store_true", help="Use text-mode result files")
    args = parser.parse_args()

    output = generate_all_tables(args.results, args.format, text_mode=args.text)
    print(output)
