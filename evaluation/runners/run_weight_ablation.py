"""Grid search over confidence weight combinations.

Runs all 30 scenarios for each weight combo and ranks by a composite
metric (ECE + Brier + taxonomy_accuracy). Produces JSON + markdown output.
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Grid parameters
WEIGHT_MIN = 0.1
WEIGHT_MAX = 0.8
WEIGHT_STEP = 0.1


def generate_weight_grid(
    step: float = WEIGHT_STEP,
    min_w: float = WEIGHT_MIN,
    max_w: float = WEIGHT_MAX,
) -> list[tuple[float, float, float]]:
    """Generate all (w_llm, w_coverage, w_base) combos that sum to 1.0.

    Each weight is in [min_w, max_w] with the given step size.

    Returns:
        List of (w_llm, w_coverage, w_base) tuples.
    """
    combos: list[tuple[float, float, float]] = []
    # Generate range values
    values = []
    w = min_w
    while w <= max_w + 1e-9:
        values.append(round(w, 2))
        w += step

    for w_llm in values:
        for w_cov in values:
            w_base = round(1.0 - w_llm - w_cov, 2)
            if min_w <= w_base <= max_w:
                combos.append((w_llm, w_cov, w_base))

    return combos


def run_weight_ablation(
    scenarios_dir: Path,
    variant: str = "S2",
    output_dir: Path | None = None,
    step: float = WEIGHT_STEP,
    min_w: float = WEIGHT_MIN,
    max_w: float = WEIGHT_MAX,
) -> dict[str, Any]:
    """Run the full weight ablation study.

    For each weight combination, runs all scenarios and computes aggregate
    metrics. Ranks combinations by composite score.

    Args:
        scenarios_dir: Path to evaluation/dataset/scenarios/.
        variant: System variant to use.
        output_dir: Where to save results.
        step: Weight step size for grid.
        min_w: Minimum weight value.
        max_w: Maximum weight value.

    Returns:
        Dict with grid results and rankings.
    """
    from evaluation.metrics.aggregate import aggregate_metrics
    from evaluation.runners.run_experiment import run_text_experiment

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = generate_weight_grid(step, min_w, max_w)
    logger.info("Weight ablation: %d combinations, variant=%s", len(grid), variant)

    results: list[dict[str, Any]] = []

    for i, (w_llm, w_cov, w_base) in enumerate(grid):
        logger.info(
            "[%d/%d] Weights: LLM=%.1f, Coverage=%.1f, Base=%.1f",
            i + 1, len(grid), w_llm, w_cov, w_base,
        )

        weight_override = {
            "CONFIDENCE_W_LLM": w_llm,
            "CONFIDENCE_W_COVERAGE": w_cov,
            "CONFIDENCE_W_BASE": w_base,
        }

        start = time.time()
        try:
            predictions, ground_truths, scenarios = run_text_experiment(
                scenarios_dir, variant, output_dir, weight_override=weight_override,
            )
            metrics = aggregate_metrics(predictions, ground_truths, scenarios=scenarios)
            elapsed = time.time() - start
        except Exception as e:
            logger.error("Failed for weights (%.1f, %.1f, %.1f): %s", w_llm, w_cov, w_base, e)
            continue

        # Composite score: higher taxonomy_accuracy + lower ECE + lower Brier
        taxonomy_acc = metrics.get("causation_accuracy_taxonomy", 0.0)
        ece = metrics.get("ece", 1.0)
        brier = metrics.get("brier_score", 1.0)
        composite = taxonomy_acc - ece - brier

        entry = {
            "w_llm": w_llm,
            "w_coverage": w_cov,
            "w_base": w_base,
            "causation_accuracy_taxonomy": taxonomy_acc,
            "ece": ece,
            "brier_score": brier,
            "composite": round(composite, 4),
            "causation_accuracy_fuzzy": metrics.get("causation_accuracy_fuzzy", 0.0),
            "factors_f1": metrics.get("factors_f1", 0.0),
            "responsibility_mae": metrics.get("responsibility_mae", 0.0),
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

    # Sort by composite score (higher is better)
    results.sort(key=lambda x: x["composite"], reverse=True)

    # Mark the default weights
    for entry in results:
        entry["is_default"] = (
            abs(entry["w_llm"] - 0.4) < 0.01
            and abs(entry["w_coverage"] - 0.3) < 0.01
            and abs(entry["w_base"] - 0.3) < 0.01
        )

    report = {
        "variant": variant,
        "n_combos": len(results),
        "grid_params": {"step": step, "min_w": min_w, "max_w": max_w},
        "rankings": results,
    }

    # Save JSON
    json_path = output_dir / "weight_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved weight ablation results to %s", json_path)

    # Save markdown table
    md_lines = [
        "# Confidence Weight Ablation Study\n",
        f"- **Variant:** {variant}",
        f"- **Combinations tested:** {len(results)}",
        f"- **Grid:** step={step}, range=[{min_w}, {max_w}]",
        "",
        "## Rankings (by composite = taxonomy_accuracy - ECE - Brier)\n",
        "| Rank | w_LLM | w_Cov | w_Base | Tax.Acc | ECE | Brier | Composite | Default? |",
        "|------|-------|-------|--------|---------|-----|-------|-----------|----------|",
    ]
    for rank, entry in enumerate(results, 1):
        default_marker = "**YES**" if entry["is_default"] else ""
        md_lines.append(
            f"| {rank} | {entry['w_llm']:.1f} | {entry['w_coverage']:.1f} | "
            f"{entry['w_base']:.1f} | {entry['causation_accuracy_taxonomy']:.3f} | "
            f"{entry['ece']:.3f} | {entry['brier_score']:.3f} | "
            f"{entry['composite']:.4f} | {default_marker} |"
        )

    # Find default rank
    default_rank = next(
        (i + 1 for i, e in enumerate(results) if e["is_default"]),
        None,
    )
    if default_rank:
        md_lines.append(f"\n**Default weights (0.4/0.3/0.3) ranked #{default_rank} out of {len(results)}.**")

    md_path = output_dir / "weight_ablation_results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Saved markdown report to %s", md_path)

    return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run confidence weight ablation study")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "scenarios",
    )
    parser.add_argument("--variant", choices=["S1", "S2", "S3"], default="S2")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--step", type=float, default=WEIGHT_STEP)
    parser.add_argument("--min-w", type=float, default=WEIGHT_MIN)
    parser.add_argument("--max-w", type=float, default=WEIGHT_MAX)
    args = parser.parse_args()

    run_weight_ablation(args.scenarios, args.variant, args.output, args.step, args.min_w, args.max_w)
