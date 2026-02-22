"""Claim-level hallucination rate measurement."""
from __future__ import annotations


def hallucination_rate(predictions: list[dict]) -> float:
    """Compute fraction of claims NOT supported by evidence.

    Each prediction should have a 'claims' list with 'supported' booleans.
    """
    total_claims = 0
    unsupported = 0

    for pred in predictions:
        claims = pred.get("claims", [])
        for claim in claims:
            total_claims += 1
            if not claim.get("supported", True):
                unsupported += 1

    return unsupported / total_claims if total_claims > 0 else 0.0


def hallucination_by_source(predictions: list[dict]) -> dict[str, float]:
    """Hallucination rate broken down by source type."""
    source_counts: dict[str, int] = {}
    source_unsupported: dict[str, int] = {}

    for pred in predictions:
        for claim in pred.get("claims", []):
            src = claim.get("source_type", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            if not claim.get("supported", True):
                source_unsupported[src] = source_unsupported.get(src, 0) + 1

    return {src: source_unsupported.get(src, 0) / count for src, count in source_counts.items() if count > 0}
