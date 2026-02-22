"""NLI-based hallucination detection using cross-encoder."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# NLI label indices for cross-encoder/nli-deberta-v3-base
_CONTRADICTION = 0
_ENTAILMENT = 1
_NEUTRAL = 2

_LABELS = ["contradiction", "entailment", "neutral"]


class NLIHallucinationChecker:
    """Detect hallucinated claims using NLI cross-encoder.

    Uses cross-encoder/nli-deberta-v3-base to classify each claim as
    entailment, contradiction, or neutral relative to the scenario premise.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.eval()
        return self._model

    def check_claim(
        self,
        premise: str,
        claim: str,
    ) -> dict[str, Any]:
        """Check a single claim against a premise using NLI.

        Returns:
            Dict with label, scores dict, and hallucinated flag.
        """
        import torch

        inputs = self._tokenizer(
            premise, claim, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        scores = {
            "contradiction": float(probs[_CONTRADICTION]),
            "entailment": float(probs[_ENTAILMENT]),
            "neutral": float(probs[_NEUTRAL]),
        }
        label_idx = int(probs.argmax())
        label = _LABELS[label_idx]

        return {
            "label": label,
            "scores": scores,
            "hallucinated": label == "contradiction",
        }

    def check_claims_batch(
        self,
        premise: str,
        claims: list[str],
    ) -> list[dict[str, Any]]:
        """Check multiple claims against a single premise."""
        return [self.check_claim(premise, claim) for claim in claims]


def nli_hallucination_rate(
    predictions: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    checker: NLIHallucinationChecker | None = None,
) -> dict[str, float]:
    """Compute NLI-based hallucination metrics across all predictions.

    For each prediction, extracts claims and checks them against the
    scenario text (or expected_claims from ground truth) using NLI.

    Args:
        predictions: List of prediction dicts with 'claims' list.
        scenarios: List of scenario dicts with 'scenario_text' and
            optionally 'ground_truth.expected_claims'.
        checker: Optional pre-initialized NLI checker.

    Returns:
        Dict with nli_hallucination_rate, nli_contradiction_rate,
        nli_entailment_rate.
    """
    if checker is None:
        checker = NLIHallucinationChecker()

    total_claims = 0
    contradictions = 0
    entailments = 0
    neutrals = 0

    for pred, scenario in zip(predictions, scenarios):
        # Build premise from scenario text
        premise = scenario.get("scenario_text", "")
        if not premise:
            gt = scenario.get("ground_truth", {})
            expected = gt.get("expected_claims", [])
            premise = " ".join(expected) if expected else ""

        if not premise:
            continue

        # Extract claims from prediction
        pred_claims = pred.get("claims", [])
        claim_texts = []
        for c in pred_claims:
            text = c.get("statement", "") or c.get("claim", "") or str(c)
            if text:
                claim_texts.append(text)

        if not claim_texts:
            # Fall back to using primary_cause as a claim
            primary = pred.get("primary_cause", "")
            if primary:
                claim_texts = [primary]

        for claim_text in claim_texts:
            result = checker.check_claim(premise, claim_text)
            total_claims += 1
            if result["label"] == "contradiction":
                contradictions += 1
            elif result["label"] == "entailment":
                entailments += 1
            else:
                neutrals += 1

    if total_claims == 0:
        return {
            "nli_hallucination_rate": 0.0,
            "nli_contradiction_rate": 0.0,
            "nli_entailment_rate": 0.0,
            "total_claims_checked": 0,
        }

    return {
        "nli_hallucination_rate": (contradictions + neutrals) / total_claims,
        "nli_contradiction_rate": contradictions / total_claims,
        "nli_entailment_rate": entailments / total_claims,
        "total_claims_checked": total_claims,
    }
