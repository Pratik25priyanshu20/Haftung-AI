"""Run N reruns per accident/scenario for entropy/stability measurement."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evaluation.metrics.hallucination_entropy import cause_stability
from evaluation.runners.run_experiment import load_scenario, run_single, run_single_text

logger = logging.getLogger(__name__)


def run_stability(
    dataset_dir: Path,
    variant: str = "S2",
    n_reruns: int = 5,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run N reruns per accident and compute stability metrics.

    Args:
        dataset_dir: Path to evaluation/dataset/accidents/.
        variant: System variant to test.
        n_reruns: Number of reruns per accident.
        output_dir: Where to save results.

    Returns:
        Dict with per-accident reruns and aggregate stability metrics.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    accident_dirs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    logger.info(
        "Stability test: %s x %d reruns x %d accidents",
        variant,
        n_reruns,
        len(accident_dirs),
    )

    rerun_results: list[list[dict[str, Any]]] = []

    for accident_dir in accident_dirs:
        logger.info("Accident %s (%d reruns)", accident_dir.name, n_reruns)
        accident_reruns: list[dict[str, Any]] = []

        for run_idx in range(n_reruns):
            logger.info("  Run %d/%d", run_idx + 1, n_reruns)
            pred = run_single(accident_dir, variant)
            accident_reruns.append(pred)

        rerun_results.append(accident_reruns)

    # Compute stability metrics
    stability = cause_stability(rerun_results)

    result = {
        "variant": variant,
        "n_reruns": n_reruns,
        "n_accidents": len(accident_dirs),
        "rerun_results": rerun_results,
        "stability_metrics": stability,
    }

    # Save
    result_path = output_dir / f"{variant}_stability_{n_reruns}runs.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        "Stability: avg_entropy=%.4f, consistency=%.4f",
        stability["avg_entropy"],
        stability["consistency_rate"],
    )
    return result


def run_text_stability(
    scenarios_dir: Path,
    variant: str = "S2",
    n_reruns: int = 5,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run N reruns per text scenario and compute stability metrics.

    Args:
        scenarios_dir: Path to evaluation/dataset/scenarios/.
        variant: System variant to test.
        n_reruns: Number of reruns per scenario.
        output_dir: Where to save results.

    Returns:
        Dict with per-scenario reruns and aggregate stability metrics.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    logger.info(
        "Text stability test: %s x %d reruns x %d scenarios",
        variant,
        n_reruns,
        len(scenario_files),
    )

    rerun_results: list[list[dict[str, Any]]] = []

    for scenario_path in scenario_files:
        scenario = load_scenario(scenario_path)
        scenario_id = scenario.get("scenario_id", scenario_path.stem)
        logger.info("Scenario %s (%d reruns)", scenario_id, n_reruns)
        scenario_reruns: list[dict[str, Any]] = []

        for run_idx in range(n_reruns):
            logger.info("  Run %d/%d", run_idx + 1, n_reruns)
            pred = run_single_text(scenario, variant)
            scenario_reruns.append(pred)

        rerun_results.append(scenario_reruns)

    # Compute stability metrics
    stability = cause_stability(rerun_results)

    result = {
        "variant": variant,
        "n_reruns": n_reruns,
        "n_scenarios": len(scenario_files),
        "rerun_results": rerun_results,
        "stability_metrics": stability,
    }

    # Save
    result_path = output_dir / f"{variant}_text_stability_{n_reruns}runs.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        "Text stability: avg_entropy=%.4f, consistency=%.4f",
        stability["avg_entropy"],
        stability["consistency_rate"],
    )
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run stability test")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--variant", choices=["S1", "S2", "S3"], default="S2")
    parser.add_argument("--reruns", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--text", action="store_true", help="Use text-only scenario mode")
    args = parser.parse_args()

    if args.text:
        run_text_stability(args.dataset, args.variant, args.reruns, args.output)
    else:
        run_stability(args.dataset, args.variant, args.reruns, args.output)
