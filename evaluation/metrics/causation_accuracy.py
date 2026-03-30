"""Primary cause match metric (legacy — use cause_taxonomy.py instead).

.. deprecated::
    This module uses substring matching which is not suitable for rigorous
    evaluation. Use ``causation_accuracy_taxonomy`` from
    ``evaluation.metrics.cause_taxonomy`` as the primary metric. This module
    is retained for backwards compatibility and is reported as
    ``causation_accuracy_fuzzy`` in aggregate results.
"""
from __future__ import annotations


def causation_accuracy(predictions: list[dict], ground_truths: list[dict]) -> float:
    """Compute fraction of predictions matching ground truth primary cause.

    Uses normalized string matching (case-insensitive, stripped).

    .. deprecated::
        Use ``causation_accuracy_taxonomy`` instead. This metric is reported
        as ``causation_accuracy_fuzzy`` in aggregate output.
    """
    if not predictions or not ground_truths:
        return 0.0

    correct = 0
    total = min(len(predictions), len(ground_truths))

    for pred, gt in zip(predictions[:total], ground_truths[:total]):
        pred_cause = pred.get("primary_cause", "").strip().lower()
        gt_cause = gt.get("primary_cause", "").strip().lower()
        if pred_cause and gt_cause and (pred_cause == gt_cause or pred_cause in gt_cause or gt_cause in pred_cause):
            correct += 1

    return correct / total if total > 0 else 0.0
