"""Calibration metrics: ECE and Brier score."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(confidences: list[float], accuracies: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: Model confidence scores [0, 1].
        accuracies: Whether prediction was correct (True/False).
        n_bins: Number of calibration bins.
    """
    if not confidences:
        return 0.0

    conf = np.array(confidences)
    acc = np.array(accuracies, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return float(ece / len(confidences))


def brier_score(confidences: list[float], accuracies: list[bool]) -> float:
    """Compute Brier score (lower is better).

    Brier = mean((confidence - accuracy)^2)
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences)
    acc = np.array(accuracies, dtype=float)
    return float(np.mean((conf - acc) ** 2))
