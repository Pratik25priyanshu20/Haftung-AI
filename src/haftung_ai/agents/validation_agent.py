"""Validation Agent — validates causation output (adapted from ARKIS)."""
from __future__ import annotations

import json
import logging
import statistics

from haftung_ai.config.settings import get_settings
from haftung_ai.llm.client import GroqClient
from haftung_ai.llm.prompts import VALIDATION_PROMPT
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)

N_JUDGE_RUNS = 3

RUBRIC_CRITERIA = ["factual_coverage", "legal_correctness", "causal_logic", "completeness"]


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

        # Multi-run LLM-based validation
        judge_result = self._llm_judge_multi(causation, chunks)
        llm_score = judge_result["mean"]

        # Evidence coverage check
        claims = causation.get("claims", [])
        supported_count = sum(1 for c in claims if c.get("supported", True))
        coverage = supported_count / max(len(claims), 1)

        # Combined confidence using configurable weights
        w_llm = self.settings.CONFIDENCE_W_LLM
        w_cov = self.settings.CONFIDENCE_W_COVERAGE
        w_base = self.settings.CONFIDENCE_W_BASE

        base_confidence = state.get("confidence_score", 0.5)
        penalty = state.get("contradiction_penalty", 0.0)
        confidence = max(0.0, min(1.0, w_llm * llm_score + w_cov * coverage + w_base * base_confidence - penalty))

        needs_correction = confidence < self.settings.VALIDATION_THRESHOLD

        state["validation_details"] = {
            "llm_score": llm_score,
            "llm_score_std": judge_result["std"],
            "llm_rubric_scores": judge_result["rubric_means"],
            "llm_rubric_stds": judge_result["rubric_stds"],
            "llm_n_runs": judge_result["n_runs"],
            "coverage": coverage,
            "base_confidence": base_confidence,
            "contradiction_penalty": penalty,
            "confidence_weights": {"w_llm": w_llm, "w_coverage": w_cov, "w_base": w_base},
            "confidence": round(confidence, 3),
            "needs_correction": needs_correction,
            "evidence_count": len(evidence),
        }
        state["confidence_score"] = round(confidence, 3)
        state["needs_correction"] = needs_correction

        logger.info(
            "ValidationAgent: confidence=%.3f (llm=%.3f±%.3f), needs_correction=%s",
            confidence, llm_score, judge_result["std"], needs_correction,
        )
        return state

    def _llm_judge_multi(self, causation: dict, chunks: list[dict]) -> dict:
        """Run the LLM judge N_JUDGE_RUNS times and aggregate results.

        Returns dict with mean, std, rubric_means, rubric_stds, n_runs.
        """
        if not chunks:
            return {
                "mean": 0.5,
                "std": 0.0,
                "rubric_means": {c: 0.5 for c in RUBRIC_CRITERIA},
                "rubric_stds": {c: 0.0 for c in RUBRIC_CRITERIA},
                "n_runs": 0,
            }

        context = "\n\n".join(f"- {c.get('content', '')}" for c in chunks[:5])
        analysis = str(causation.get("primary_cause", "")) + " " + str(causation.get("reasoning", ""))
        prompt = VALIDATION_PROMPT.format(context=context, analysis=analysis)

        all_scores: list[float] = []
        rubric_scores: dict[str, list[float]] = {c: [] for c in RUBRIC_CRITERIA}

        for run_idx in range(N_JUDGE_RUNS):
            try:
                result_text = self.llm.judge(prompt).strip()
                parsed = self._parse_rubric(result_text)
                per_criterion = [parsed.get(c, 0.5) for c in RUBRIC_CRITERIA]
                run_mean = statistics.mean(per_criterion)
                all_scores.append(run_mean)
                for c in RUBRIC_CRITERIA:
                    rubric_scores[c].append(parsed.get(c, 0.5))
            except Exception as e:
                logger.error("Validation LLM judge run %d failed: %s", run_idx, e)
                all_scores.append(0.5)
                for c in RUBRIC_CRITERIA:
                    rubric_scores[c].append(0.5)

        mean_score = statistics.mean(all_scores)
        std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        rubric_means = {c: statistics.mean(rubric_scores[c]) for c in RUBRIC_CRITERIA}
        rubric_stds = {
            c: statistics.stdev(rubric_scores[c]) if len(rubric_scores[c]) > 1 else 0.0
            for c in RUBRIC_CRITERIA
        }

        return {
            "mean": round(mean_score, 4),
            "std": round(std_score, 4),
            "rubric_means": {c: round(v, 4) for c, v in rubric_means.items()},
            "rubric_stds": {c: round(v, 4) for c, v in rubric_stds.items()},
            "n_runs": len(all_scores),
        }

    @staticmethod
    def _parse_rubric(text: str) -> dict[str, float]:
        """Parse JSON rubric response from the LLM judge."""
        # Try to extract JSON from the response
        try:
            # Handle case where LLM wraps JSON in markdown code blocks
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(cleaned)
            return {
                k: max(0.0, min(1.0, float(v)))
                for k, v in parsed.items()
                if k in RUBRIC_CRITERIA
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Could not parse rubric JSON, falling back to 0.5")
            return {c: 0.5 for c in RUBRIC_CRITERIA}
