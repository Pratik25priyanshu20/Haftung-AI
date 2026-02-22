"""Validation Agent — validates causation output (adapted from ARKIS)."""
from __future__ import annotations

import logging
import re

from haftung_ai.config.settings import get_settings
from haftung_ai.llm.client import GroqClient
from haftung_ai.llm.prompts import VALIDATION_PROMPT
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)


class ValidationAgent:
    """Validates causation analysis against evidence. Produces confidence score."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = GroqClient(temperature=0.0)

    def __call__(self, state: HaftungState) -> HaftungState:
        causation = state.get("causation_output", {})
        chunks = state.get("retrieved_chunks", [])
        evidence = state.get("evidence", [])

        if not causation:
            state["validation_details"] = {"confidence": 0.0, "needs_correction": True}
            state["confidence_score"] = 0.0
            state["needs_correction"] = True
            return state

        # LLM-based validation
        llm_score = self._llm_judge(causation, chunks)

        # Evidence coverage check
        claims = causation.get("claims", [])
        supported_count = sum(1 for c in claims if c.get("supported", True))
        coverage = supported_count / max(len(claims), 1)

        # Combined confidence
        base_confidence = state.get("confidence_score", 0.5)
        penalty = state.get("contradiction_penalty", 0.0)
        confidence = max(0.0, min(1.0, 0.4 * llm_score + 0.3 * coverage + 0.3 * base_confidence - penalty))

        needs_correction = confidence < self.settings.VALIDATION_THRESHOLD

        state["validation_details"] = {
            "llm_score": llm_score,
            "coverage": coverage,
            "base_confidence": base_confidence,
            "contradiction_penalty": penalty,
            "confidence": round(confidence, 3),
            "needs_correction": needs_correction,
            "evidence_count": len(evidence),
        }
        state["confidence_score"] = round(confidence, 3)
        state["needs_correction"] = needs_correction

        logger.info("ValidationAgent: confidence=%.3f, needs_correction=%s", confidence, needs_correction)
        return state

    def _llm_judge(self, causation: dict, chunks: list[dict]) -> float:
        if not chunks:
            return 0.5  # No evidence to validate against
        context = "\n\n".join(f"- {c.get('content', '')}" for c in chunks[:5])
        analysis = str(causation.get("primary_cause", "")) + " " + str(causation.get("reasoning", ""))
        prompt = VALIDATION_PROMPT.format(context=context, analysis=analysis)

        try:
            result = self.llm.judge(prompt).strip()
            match = re.search(r"(\d+(?:\.\d+)?)", result)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
        except Exception as e:
            logger.error("Validation LLM judge failed: %s", e)
        return 0.5
