"""Dedicated runner for LLM judge variance analysis.

Runs the validation judge N times per scenario to quantify inter-run
variance. Produces a JSON report with per-scenario and aggregate
mean/std statistics.
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from haftung_ai.agents.orchestrator import build_text_graph
from haftung_ai.agents.validation_agent import RUBRIC_CRITERIA

logger = logging.getLogger(__name__)

DEFAULT_N_RUNS = 5


def run_judge_variance(
    scenarios_dir: Path,
    variant: str = "S2",
    n_runs: int = DEFAULT_N_RUNS,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the validation judge multiple times per scenario.

    For each scenario, the full pipeline is run once to get the causation
    output and retrieved chunks. Then the judge is re-run n_runs times
    to measure variance.

    Args:
        scenarios_dir: Path to evaluation/dataset/scenarios/.
        variant: System variant to use.
        n_runs: Number of judge runs per scenario.
        output_dir: Where to save results. Defaults to evaluation/results/.

    Returns:
        Dict with per-scenario and aggregate variance statistics.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    logger.info("Judge variance analysis: %d scenarios, %d runs each", len(scenario_files), n_runs)

    per_scenario: list[dict[str, Any]] = []
    all_stds: list[float] = []
    rubric_all_stds: dict[str, list[float]] = {c: [] for c in RUBRIC_CRITERIA}

    for scenario_path in scenario_files:
        with open(scenario_path) as f:
            scenario = json.load(f)

        scenario_id = scenario.get("scenario_id", scenario_path.stem)
        logger.info("Processing %s ...", scenario_id)

        # Run the full pipeline once
        graph = build_text_graph(variant)
        initial_state = {
            "scenario_text": scenario.get("scenario_text", ""),
            "variant": variant,
            "errors": [],
            "warnings": [],
        }

        try:
            result = graph.invoke(initial_state)
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", scenario_id, e)
            continue

        # Extract what we need for re-running the judge
        causation = result.get("causation_output", {})
        chunks = result.get("retrieved_chunks", [])

        if not causation:
            logger.warning("No causation output for %s, skipping", scenario_id)
            continue

        # Re-run the judge n_runs times
        from haftung_ai.agents.validation_agent import ValidationAgent
        validator = ValidationAgent()

        run_scores: list[float] = []
        run_rubrics: dict[str, list[float]] = {c: [] for c in RUBRIC_CRITERIA}

        for run_idx in range(n_runs):
            judge_result = validator._llm_judge_multi(causation, chunks)
            run_scores.append(judge_result["mean"])
            for c in RUBRIC_CRITERIA:
                run_rubrics[c].append(judge_result["rubric_means"].get(c, 0.5))

        mean_score = statistics.mean(run_scores)
        std_score = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
        all_stds.append(std_score)

        rubric_stats = {}
        for c in RUBRIC_CRITERIA:
            c_mean = statistics.mean(run_rubrics[c])
            c_std = statistics.stdev(run_rubrics[c]) if len(run_rubrics[c]) > 1 else 0.0
            rubric_stats[c] = {"mean": round(c_mean, 4), "std": round(c_std, 4)}
            rubric_all_stds[c].append(c_std)

        per_scenario.append({
            "scenario_id": scenario_id,
            "n_runs": n_runs,
            "judge_mean": round(mean_score, 4),
            "judge_std": round(std_score, 4),
            "judge_scores": [round(s, 4) for s in run_scores],
            "rubric_stats": rubric_stats,
        })

    # Aggregate
    aggregate = {
        "n_scenarios": len(per_scenario),
        "n_runs_per_scenario": n_runs,
        "variant": variant,
        "mean_judge_std": round(statistics.mean(all_stds), 4) if all_stds else 0.0,
        "max_judge_std": round(max(all_stds), 4) if all_stds else 0.0,
        "rubric_mean_stds": {
            c: round(statistics.mean(rubric_all_stds[c]), 4) if rubric_all_stds[c] else 0.0
            for c in RUBRIC_CRITERIA
        },
    }

    report = {
        "aggregate": aggregate,
        "per_scenario": per_scenario,
    }

    # Save
    output_path = output_dir / "judge_variance_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved judge variance report to %s", output_path)

    # Also save a readable markdown summary
    md_lines = [
        "# LLM Judge Variance Report\n",
        f"- **Variant:** {variant}",
        f"- **Scenarios:** {aggregate['n_scenarios']}",
        f"- **Runs per scenario:** {n_runs}",
        f"- **Mean cross-run std:** {aggregate['mean_judge_std']:.4f}",
        f"- **Max cross-run std:** {aggregate['max_judge_std']:.4f}",
        "",
        "## Per-Rubric Mean Std\n",
    ]
    for c in RUBRIC_CRITERIA:
        md_lines.append(f"- **{c}:** {aggregate['rubric_mean_stds'][c]:.4f}")
    md_lines.append("")
    md_lines.append("## Per-Scenario Details\n")
    md_lines.append("| Scenario | Mean | Std | Scores |")
    md_lines.append("|---|---|---|---|")
    for entry in per_scenario:
        scores_str = ", ".join(f"{s:.3f}" for s in entry["judge_scores"])
        md_lines.append(
            f"| {entry['scenario_id']} | {entry['judge_mean']:.4f} "
            f"| {entry['judge_std']:.4f} | {scores_str} |"
        )

    md_path = output_dir / "judge_variance_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Saved markdown report to %s", md_path)

    return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run LLM judge variance analysis")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "scenarios",
    )
    parser.add_argument("--variant", choices=["S1", "S2", "S3"], default="S2")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_judge_variance(args.scenarios, args.variant, args.n_runs, args.output)
