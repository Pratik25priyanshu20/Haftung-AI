"""Aggregate all metrics into a single report."""
from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics.calibration import brier_score, expected_calibration_error
from evaluation.metrics.causation_accuracy import causation_accuracy
from evaluation.metrics.cause_taxonomy import causation_accuracy_taxonomy
from evaluation.metrics.factors_f1 import factors_f1
from evaluation.metrics.hallucination import hallucination_rate
from evaluation.metrics.hallucination_entropy import cause_stability
from evaluation.metrics.responsibility_mae import responsibility_mae
from evaluation.metrics.retrieval_quality import retrieval_quality_metrics

logger = logging.getLogger(__name__)


def aggregate_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    rerun_results: list[list[dict]] | None = None,
    scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute all metrics and return aggregated results.

    Args:
        predictions: List of prediction dicts from the pipeline.
        ground_truths: List of ground truth dicts.
        rerun_results: Optional rerun groups for stability metrics.
        scenarios: Optional scenario dicts for NLI hallucination checking.
    """
    results: dict[str, Any] = {}

    # --- Legacy fuzzy string matching (kept for backwards compatibility) ---
    results["causation_accuracy_fuzzy"] = causation_accuracy(predictions, ground_truths)
    results["responsibility_mae"] = responsibility_mae(predictions, ground_truths)

    f1_result = factors_f1(predictions, ground_truths)
    results["factors_precision"] = f1_result["precision"]
    results["factors_recall"] = f1_result["recall"]
    results["factors_f1"] = f1_result["f1"]

    results["hallucination_rate"] = hallucination_rate(predictions)

    # --- Taxonomy-based causation accuracy [PRIMARY] ---
    taxonomy_result = causation_accuracy_taxonomy(predictions, ground_truths)
    results["causation_accuracy_taxonomy"] = taxonomy_result["exact_match"]
    results["causation_per_category"] = taxonomy_result["per_category"]

    # --- Calibration (ECE / Brier) using taxonomy match as correctness ---
    confidences = [p.get("confidence", 0.5) for p in predictions]
    taxonomy_accuracies = taxonomy_result["per_item"]
    results["ece"] = expected_calibration_error(confidences, taxonomy_accuracies)
    results["brier_score"] = brier_score(confidences, taxonomy_accuracies)

    # --- Retrieval quality (S2/S3 only) ---
    has_retrieval = any(p.get("retrieved_chunks") for p in predictions)
    if has_retrieval:
        retrieval = retrieval_quality_metrics(predictions, ground_truths, k=5)
        results["precision_at_5"] = retrieval["precision_at_5"]
        results["mrr"] = retrieval["mrr"]
        results["ndcg_at_5"] = retrieval["ndcg_at_5"]

    # --- NLI hallucination (if scenarios provided) ---
    if scenarios:
        try:
            from evaluation.metrics.nli_hallucination import nli_hallucination_rate

            nli_result = nli_hallucination_rate(predictions, scenarios)
            results["nli_hallucination_rate"] = nli_result["nli_hallucination_rate"]
            results["nli_contradiction_rate"] = nli_result["nli_contradiction_rate"]
            results["nli_entailment_rate"] = nli_result["nli_entailment_rate"]
        except Exception as e:
            logger.warning("NLI hallucination check failed: %s", e)

    # --- Latency ---
    latencies = [p.get("elapsed_seconds", 0.0) for p in predictions if p.get("elapsed_seconds")]
    if latencies:
        results["mean_latency_s"] = sum(latencies) / len(latencies)

    retrieval_latencies = [
        p.get("retrieval_latency_s", 0.0) for p in predictions if p.get("retrieval_latency_s")
    ]
    if retrieval_latencies:
        results["mean_retrieval_latency_s"] = sum(retrieval_latencies) / len(retrieval_latencies)

    # --- Stability (rerun) ---
    if rerun_results:
        stability = cause_stability(rerun_results)
        results["avg_entropy"] = stability["avg_entropy"]
        results["consistency_rate"] = stability["consistency_rate"]

    # --- Metadata ---
    results["_metric_notes"] = {
        "causation_accuracy_taxonomy": "[PRIMARY] Taxonomy-based cause matching via keyword classifier (cause_taxonomy.py).",
        "causation_accuracy_fuzzy": "[legacy] Substring-based string matching. Retained for backwards compatibility only.",
        "ece": "Expected Calibration Error computed against taxonomy match correctness.",
        "brier_score": "Brier score computed against taxonomy match correctness.",
    }
    results["_synthetic_data_notice"] = (
        "All telemetry and scenario data used in this evaluation is fully "
        "synthetic. CAN bus logs are generated from parametric models "
        "(scripts/generate_synthetic_can.py). Text scenarios are authored "
        "templates. No real accident data is used. See README.md § Limitations."
    )

    return results
