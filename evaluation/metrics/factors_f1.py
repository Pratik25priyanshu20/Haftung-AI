"""Contributing factors F1 score."""
from __future__ import annotations


def factors_f1(predictions: list[dict], ground_truths: list[dict]) -> dict[str, float]:
    """Compute precision, recall, F1 for contributing factors.

    Matches factors by normalized category string.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_factors = {f.get("category", "").lower() for f in pred.get("contributing_factors", [])}
        gt_factors = {f.get("category", "").lower() for f in gt.get("contributing_factors", [])}

        tp = len(pred_factors & gt_factors)
        fp = len(pred_factors - gt_factors)
        fn = len(gt_factors - pred_factors)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
