"""Stability measurement via entropy across reruns."""
from __future__ import annotations

import math
from collections import Counter


def answer_entropy(answers: list[str]) -> float:
    """Compute Shannon entropy of answer distribution across reruns.

    Lower entropy = more stable/consistent.
    """
    if not answers:
        return 0.0

    counts = Counter(answers)
    total = len(answers)
    entropy = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def cause_stability(rerun_results: list[list[dict]]) -> dict[str, float]:
    """Compute stability metrics across reruns for each accident.

    Args:
        rerun_results: List of rerun groups, each containing N predictions for the same accident.

    Returns:
        Dict with avg_entropy and consistency_rate.
    """
    entropies: list[float] = []
    consistent = 0

    for reruns in rerun_results:
        causes = [r.get("primary_cause", "") for r in reruns]
        e = answer_entropy(causes)
        entropies.append(e)
        if e == 0.0 and len(set(causes)) == 1:
            consistent += 1

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    consistency = consistent / len(rerun_results) if rerun_results else 0.0

    return {"avg_entropy": avg_entropy, "consistency_rate": consistency}
