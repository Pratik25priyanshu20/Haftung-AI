"""Responsibility percentage deviation (MAE)."""
from __future__ import annotations


def responsibility_mae(predictions: list[dict], ground_truths: list[dict]) -> float:
    """Mean absolute error of responsibility percentages.

    Matches parties by name, computes MAE across all party-accident pairs.
    """
    errors: list[float] = []

    for pred, gt in zip(predictions, ground_truths):
        pred_resp = {r["party"]: r["percentage"] for r in pred.get("responsibility", [])}
        gt_resp = {r["party"]: r["percentage"] for r in gt.get("responsibility", [])}

        all_parties = set(pred_resp.keys()) | set(gt_resp.keys())
        for party in all_parties:
            p = pred_resp.get(party, 0.0)
            g = gt_resp.get(party, 0.0)
            errors.append(abs(p - g))

    return sum(errors) / len(errors) if errors else 0.0
