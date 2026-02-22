"""Run all 3 system variants and compute comparative metrics."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evaluation.metrics.aggregate import aggregate_metrics
from evaluation.runners.run_experiment import (
    load_ground_truth,
    run_experiment,
    run_text_experiment,
)

logger = logging.getLogger(__name__)


def run_all_systems(
    dataset_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Run S1, S2, S3 and return comparative metrics.

    Returns:
        Dict mapping variant name to its metric dict.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect ground truths once
    accident_dirs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    ground_truths = [load_ground_truth(d) for d in accident_dirs]

    all_metrics: dict[str, dict[str, float]] = {}

    for variant in ("S1", "S2", "S3"):
        logger.info("=== Running variant %s ===", variant)
        predictions = run_experiment(dataset_dir, variant, output_dir)

        metrics = aggregate_metrics(predictions, ground_truths)
        all_metrics[variant] = metrics

        logger.info("Metrics for %s: %s", variant, metrics)

    # Save comparative summary
    summary_path = output_dir / "comparative_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Comparative metrics saved to %s", summary_path)
    return all_metrics


def run_all_text_systems(
    scenarios_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run S1, S2, S3 in text-only mode and return comparative metrics.

    Args:
        scenarios_dir: Path to evaluation/dataset/scenarios/.
        output_dir: Where to save results JSON.

    Returns:
        Dict mapping variant name to its metric dict.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, Any]] = {}

    for variant in ("S1", "S2", "S3"):
        logger.info("=== Running text variant %s ===", variant)
        predictions, ground_truths, scenarios = run_text_experiment(
            scenarios_dir, variant, output_dir
        )

        metrics = aggregate_metrics(
            predictions, ground_truths, scenarios=scenarios
        )
        all_metrics[variant] = metrics

        logger.info("Metrics for %s: %s", variant, metrics)

    # Save comparative summary
    summary_path = output_dir / "comparative_text_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Comparative text metrics saved to %s", summary_path)
    return all_metrics


def load_or_run(
    dataset_dir: Path,
    output_dir: Path,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load cached results or run experiment."""
    pred_path = output_dir / f"{variant}_predictions.json"
    gt_path = output_dir / f"{variant}_ground_truths.json"

    if pred_path.exists() and gt_path.exists():
        logger.info("Loading cached results for %s", variant)
        with open(pred_path) as f:
            predictions = json.load(f)
        with open(gt_path) as f:
            ground_truths = json.load(f)
        return predictions, ground_truths

    predictions = run_experiment(dataset_dir, variant, output_dir)
    accident_dirs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    ground_truths = [load_ground_truth(d) for d in accident_dirs]
    return predictions, ground_truths


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run all system variants")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--text", action="store_true", help="Use text-only scenario mode")
    args = parser.parse_args()

    if args.text:
        run_all_text_systems(args.dataset, args.output)
    else:
        run_all_systems(args.dataset, args.output)
