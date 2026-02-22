"""Retrieval quality metrics: Precision@k, MRR, nDCG@k."""
from __future__ import annotations

import math
from typing import Any


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Precision@k: fraction of top-k retrieved items that are relevant.

    Args:
        retrieved: Ordered list of retrieved chunk IDs / references.
        relevant: Set of ground-truth relevant IDs / references.
        k: Cutoff.
    """
    if not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item.

    Args:
        retrieved: Ordered list of retrieved chunk IDs / references.
        relevant: Set of ground-truth relevant IDs / references.
    """
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 if in relevant set, 0 otherwise.

    Args:
        retrieved: Ordered list of retrieved chunk IDs / references.
        relevant: Set of ground-truth relevant IDs / references.
        k: Cutoff.
    """
    if not retrieved or not relevant:
        return 0.0

    top_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant items ranked first
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def _extract_stvo_refs(chunk: dict[str, Any]) -> list[str]:
    """Extract StVO paragraph references from a retrieved chunk.

    Looks in chunk_id, content, and metadata for §-references.
    """
    import re

    refs: set[str] = set()
    # Check chunk_id (e.g. "stvo_abstand" -> "§4 StVO" mapping)
    chunk_id = chunk.get("chunk_id", "")
    content = chunk.get("content", "")
    metadata = chunk.get("metadata", {})

    # Search for §X StVO patterns in content
    text = f"{chunk_id} {content} {metadata.get('source', '')} {metadata.get('section', '')}"
    pattern = r"§\s*\d+(?:\s*(?:Abs\.\s*\d+)?)\s*StVO"
    for match in re.findall(pattern, text):
        # Normalize: remove extra spaces
        normalized = re.sub(r"\s+", " ", match).strip()
        refs.add(normalized)

    # Also check for chunk_id-based mapping
    stvo_map = {
        "stvo_abstand": "§4 StVO",
        "stvo_geschwindigkeit": "§3 StVO",
        "stvo_vorfahrt": "§8 StVO",
        "stvo_ueberholen": "§5 StVO",
        "stvo_fussgaenger": "§26 StVO",
        "stvo_abbiegen": "§9 StVO",
        "stvo_lichtzeichen": "§37 StVO",
    }
    for key, ref in stvo_map.items():
        if key in chunk_id.lower():
            refs.add(ref)

    return list(refs)


def retrieval_quality_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, float]:
    """Compute retrieval quality metrics across all predictions.

    Matches retrieved chunks against `relevant_stvo` in ground truth
    by extracting StVO references from retrieved chunks.

    Args:
        predictions: List of prediction dicts with 'retrieved_chunks'.
        ground_truths: List of ground truth dicts with 'relevant_stvo'.
        k: Cutoff for Precision@k and nDCG@k.

    Returns:
        Dict with precision_at_k, mrr, ndcg_at_k averaged across samples.
    """
    precisions: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    for pred, gt in zip(predictions, ground_truths):
        chunks = pred.get("retrieved_chunks", [])
        gt_data = gt.get("ground_truth", gt)
        relevant_stvo = set(gt_data.get("relevant_stvo", []))

        if not relevant_stvo:
            continue

        # Extract StVO references from each retrieved chunk
        retrieved_refs: list[str] = []
        for chunk in chunks:
            refs = _extract_stvo_refs(chunk)
            retrieved_refs.extend(refs)

        # Normalize ground truth references
        import re
        normalized_relevant = set()
        for ref in relevant_stvo:
            normalized = re.sub(r"\s+", " ", ref).strip()
            normalized_relevant.add(normalized)

        precisions.append(precision_at_k(retrieved_refs, normalized_relevant, k))
        mrrs.append(mean_reciprocal_rank(retrieved_refs, normalized_relevant))
        ndcgs.append(ndcg_at_k(retrieved_refs, normalized_relevant, k))

    n = len(precisions)
    return {
        f"precision_at_{k}": sum(precisions) / n if n > 0 else 0.0,
        "mrr": sum(mrrs) / n if n > 0 else 0.0,
        f"ndcg_at_{k}": sum(ndcgs) / n if n > 0 else 0.0,
        "n_evaluated": n,
    }
