"""Run a single experiment: one system variant against the eval dataset."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from haftung_ai.agents.orchestrator import build_graph, build_text_graph
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)


def load_ground_truth(accident_dir: Path) -> dict[str, Any]:
    """Load ground truth JSON from an accident directory."""
    gt_path = accident_dir / "ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def load_scenario(scenario_path: Path) -> dict[str, Any]:
    """Load a text scenario JSON file."""
    with open(scenario_path) as f:
        return json.load(f)


def run_single(
    accident_dir: Path,
    variant: str,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run one variant on one accident, return prediction dict."""
    video_path = accident_dir / "video.mp4"
    can_log_path = accident_dir / "can_log.csv"

    graph = build_graph(variant)
    initial_state: HaftungState = {
        "video_path": str(video_path),
        "can_log_path": str(can_log_path),
        "variant": variant,
        "errors": [],
        "warnings": [],
    }

    start = time.time()
    try:
        result = graph.invoke(initial_state)
        elapsed = time.time() - start
    except Exception as e:
        logger.error("Failed on %s with %s: %s", accident_dir.name, variant, e)
        return {
            "accident_id": accident_dir.name,
            "variant": variant,
            "error": str(e),
            "elapsed_seconds": time.time() - start,
        }

    return {
        "accident_id": accident_dir.name,
        "variant": variant,
        "primary_cause": result.get("primary_cause", ""),
        "accident_type": result.get("accident_type", ""),
        "responsibility": result.get("responsibility", []),
        "contributing_factors": result.get("contributing_factors", []),
        "claims": result.get("claims", []),
        "confidence": result.get("confidence_score"),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "retrieval_latency_s": result.get("retrieval_latency_s"),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "elapsed_seconds": elapsed,
    }


def run_single_text(
    scenario: dict[str, Any],
    variant: str,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run one variant on one text scenario, return prediction dict."""
    scenario_text = scenario.get("scenario_text", "")
    scenario_id = scenario.get("scenario_id", "unknown")

    graph = build_text_graph(variant)
    initial_state: HaftungState = {
        "scenario_text": scenario_text,
        "variant": variant,
        "errors": [],
        "warnings": [],
    }

    start = time.time()
    try:
        result = graph.invoke(initial_state)
        elapsed = time.time() - start
    except Exception as e:
        logger.error("Failed on %s with %s: %s", scenario_id, variant, e)
        return {
            "scenario_id": scenario_id,
            "variant": variant,
            "error": str(e),
            "elapsed_seconds": time.time() - start,
        }

    return {
        "scenario_id": scenario_id,
        "variant": variant,
        "primary_cause": result.get("primary_cause", ""),
        "accident_type": result.get("accident_type", ""),
        "responsibility": result.get("responsibility", []),
        "contributing_factors": result.get("contributing_factors", []),
        "claims": result.get("claims", []),
        "confidence": result.get("confidence_score"),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "retrieval_latency_s": result.get("retrieval_latency_s"),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "elapsed_seconds": elapsed,
    }


def run_experiment(
    dataset_dir: Path,
    variant: str,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run a variant against all accidents in the dataset.

    Args:
        dataset_dir: Path to evaluation/dataset/accidents/.
        variant: System variant (S1, S2, S3).
        output_dir: Where to save results JSON. Defaults to evaluation/results/.

    Returns:
        List of prediction dicts.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    accident_dirs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    logger.info("Running %s on %d accidents", variant, len(accident_dirs))

    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []

    for accident_dir in accident_dirs:
        logger.info("Processing %s ...", accident_dir.name)
        pred = run_single(accident_dir, variant)
        predictions.append(pred)

        gt = load_ground_truth(accident_dir)
        ground_truths.append(gt)

    # Save raw results
    result_path = output_dir / f"{variant}_predictions.json"
    with open(result_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    gt_path = output_dir / f"{variant}_ground_truths.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truths, f, indent=2, default=str)

    logger.info("Saved results to %s", result_path)
    return predictions


def run_text_experiment(
    scenarios_dir: Path,
    variant: str,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a variant against all text scenarios.

    Args:
        scenarios_dir: Path to evaluation/dataset/scenarios/.
        variant: System variant (S1, S2, S3).
        output_dir: Where to save results JSON. Defaults to evaluation/results/.

    Returns:
        Tuple of (predictions, ground_truths, scenarios).
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    logger.info("Running %s on %d text scenarios", variant, len(scenario_files))

    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []

    for scenario_path in scenario_files:
        scenario = load_scenario(scenario_path)
        scenarios.append(scenario)
        logger.info("Processing %s ...", scenario.get("scenario_id", scenario_path.stem))

        pred = run_single_text(scenario, variant)
        predictions.append(pred)

        gt = scenario.get("ground_truth", {})
        ground_truths.append(gt)

    # Save raw results
    result_path = output_dir / f"{variant}_text_predictions.json"
    with open(result_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    gt_path = output_dir / f"{variant}_text_ground_truths.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truths, f, indent=2, default=str)

    scenarios_path = output_dir / f"{variant}_text_scenarios.json"
    with open(scenarios_path, "w") as f:
        json.dump(scenarios, f, indent=2, default=str)

    logger.info("Saved text results to %s", result_path)
    return predictions, ground_truths, scenarios


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run single experiment")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to accidents or scenarios dir")
    parser.add_argument("--variant", choices=["S1", "S2", "S3"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--text", action="store_true", help="Use text-only scenario mode")
    args = parser.parse_args()

    if args.text:
        run_text_experiment(args.dataset, args.variant, args.output)
    else:
        run_experiment(args.dataset, args.variant, args.output)
