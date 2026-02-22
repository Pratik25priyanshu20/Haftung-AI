"""Compare metrics across S1, S2, S3 system variants."""
from __future__ import annotations

import json
from pathlib import Path


def load_comparative_metrics(
    results_dir: Path, text_mode: bool = False
) -> dict[str, dict[str, float]]:
    """Load the comparative metrics JSON from a results directory."""
    filename = "comparative_text_metrics.json" if text_mode else "comparative_metrics.json"
    path = results_dir / filename
    with open(path) as f:
        return json.load(f)


def compute_deltas(
    metrics: dict[str, dict[str, float]],
    baseline: str = "S1",
) -> dict[str, dict[str, float]]:
    """Compute metric deltas relative to a baseline variant.

    Positive delta = improvement for metrics where higher is better.
    For MAE/ECE/Brier/hallucination_rate/avg_entropy, sign is flipped so
    positive still means improvement.
    """
    lower_is_better = {
        "responsibility_mae",
        "ece",
        "brier_score",
        "hallucination_rate",
        "avg_entropy",
    }

    base = metrics[baseline]
    deltas: dict[str, dict[str, float]] = {}

    for variant, variant_metrics in metrics.items():
        if variant == baseline:
            continue
        d: dict[str, float] = {}
        for metric_name, value in variant_metrics.items():
            base_val = base.get(metric_name, 0.0)
            raw_delta = value - base_val
            if metric_name in lower_is_better:
                d[metric_name] = -raw_delta  # flip so positive = better
            else:
                d[metric_name] = raw_delta
        deltas[variant] = d

    return deltas


def format_comparison_table(metrics: dict[str, dict[str, float]]) -> str:
    """Format metrics as a markdown comparison table."""
    if not metrics:
        return "No metrics to compare."

    # Collect all metric names
    all_metrics: set[str] = set()
    for variant_metrics in metrics.values():
        all_metrics.update(variant_metrics.keys())
    metric_names = sorted(all_metrics)

    # Header
    variants = sorted(metrics.keys())
    header = "| Metric | " + " | ".join(variants) + " |"
    separator = "|---|" + "|".join(["---"] * len(variants)) + "|"

    rows = [header, separator]
    for metric in metric_names:
        values = []
        for variant in variants:
            val = metrics[variant].get(metric)
            values.append(f"{val:.4f}" if val is not None else "N/A")
        rows.append(f"| {metric} | " + " | ".join(values) + " |")

    return "\n".join(rows)


def rank_variants(metrics: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Rank variants per metric (best first).

    Returns dict mapping metric name to list of variant names, sorted best-first.
    """
    lower_is_better = {
        "responsibility_mae",
        "ece",
        "brier_score",
        "hallucination_rate",
        "avg_entropy",
    }

    rankings: dict[str, list[str]] = {}
    all_metrics: set[str] = set()
    for vm in metrics.values():
        all_metrics.update(vm.keys())

    for metric_name in all_metrics:
        variant_vals: list[tuple[str, float]] = []
        for variant, vm in metrics.items():
            if metric_name in vm:
                variant_vals.append((variant, vm[metric_name]))

        reverse = metric_name not in lower_is_better
        variant_vals.sort(key=lambda x: x[1], reverse=reverse)
        rankings[metric_name] = [v[0] for v in variant_vals]

    return rankings


def generate_report(results_dir: Path, text_mode: bool = False) -> str:
    """Generate a full comparison report as markdown."""
    metrics = load_comparative_metrics(results_dir, text_mode=text_mode)
    deltas = compute_deltas(metrics)
    rankings = rank_variants(metrics)

    lines: list[str] = []
    lines.append("# System Comparison Report\n")
    lines.append("## Metrics\n")
    lines.append(format_comparison_table(metrics))
    lines.append("")

    lines.append("## Rankings (best first)\n")
    for metric, ranked in sorted(rankings.items()):
        lines.append(f"- **{metric}**: {' > '.join(ranked)}")

    lines.append("")
    lines.append("## Deltas vs S1 (positive = improvement)\n")
    for variant, delta_metrics in sorted(deltas.items()):
        lines.append(f"\n### {variant} vs S1\n")
        for metric, val in sorted(delta_metrics.items()):
            sign = "+" if val >= 0 else ""
            lines.append(f"- {metric}: {sign}{val:.4f}")

    return "\n".join(lines)


def per_category_comparison(
    results_dir: Path,
    text_mode: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute metrics broken down by accident category.

    Returns:
        Dict mapping category -> variant -> metric -> value.
    """
    from evaluation.metrics.cause_taxonomy import classify_cause

    variants = ["S1", "S2", "S3"]
    pred_suffix = "_text_predictions.json" if text_mode else "_predictions.json"
    gt_suffix = "_text_ground_truths.json" if text_mode else "_ground_truths.json"

    # Load all variant results
    variant_data: dict[str, tuple[list[dict], list[dict]]] = {}
    for variant in variants:
        pred_path = results_dir / f"{variant}{pred_suffix}"
        gt_path = results_dir / f"{variant}{gt_suffix}"
        if not pred_path.exists():
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        with open(gt_path) as f:
            gts = json.load(f)
        variant_data[variant] = (preds, gts)

    if not variant_data:
        return {}

    # Group by category
    categories = ["rear_end", "side_collision", "head_on", "intersection", "pedestrian", "single_vehicle"]
    result: dict[str, dict[str, dict[str, float]]] = {}

    for variant, (preds, gts) in variant_data.items():
        for category in categories:
            if category not in result:
                result[category] = {}

            cat_preds = []
            cat_gts = []
            for p, g in zip(preds, gts):
                gt_cat = g.get("category", g.get("accident_type", ""))
                if gt_cat == category:
                    cat_preds.append(p)
                    cat_gts.append(g)

            if not cat_preds:
                continue

            # Compute per-category metrics
            correct = sum(
                1 for p, g in zip(cat_preds, cat_gts)
                if classify_cause(p.get("primary_cause", "")) == g.get("primary_cause_taxonomy_id", "")
            )
            accuracy = correct / len(cat_preds)

            resp_errors = []
            for p, g in zip(cat_preds, cat_gts):
                pred_resp = {r.get("party", ""): r.get("percentage", 0.0) for r in p.get("responsibility", [])}
                gt_resp = {r.get("party", ""): r.get("percentage", 0.0) for r in g.get("responsibility", [])}
                parties = set(pred_resp) | set(gt_resp)
                if parties:
                    resp_errors.append(
                        sum(abs(pred_resp.get(x, 0) - gt_resp.get(x, 0)) for x in parties) / len(parties)
                    )

            result[category][variant] = {
                "accuracy": accuracy,
                "n": len(cat_preds),
                "responsibility_mae": sum(resp_errors) / len(resp_errors) if resp_errors else 0.0,
            }

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare system variants")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--text", action="store_true", help="Use text-mode result files")
    args = parser.parse_args()

    report = generate_report(args.results, text_mode=args.text)
    print(report)

    suffix = "_text" if args.text else ""
    output_path = args.results / f"comparison_report{suffix}.md"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nSaved to {output_path}")

    if args.text:
        per_cat = per_category_comparison(args.results, text_mode=True)
        if per_cat:
            print("\n## Per-Category Comparison\n")
            for cat, variants in sorted(per_cat.items()):
                print(f"\n### {cat}")
                for variant, m in sorted(variants.items()):
                    print(f"  {variant}: accuracy={m['accuracy']:.3f}, mae={m['responsibility_mae']:.1f}, n={m['n']}")

            cat_path = args.results / "per_category_comparison.json"
            with open(cat_path, "w") as f:
                json.dump(per_cat, f, indent=2)
            print(f"\nSaved to {cat_path}")
