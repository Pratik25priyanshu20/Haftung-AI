"""Contradiction Agent — detects contradictions in evidence (adapted from ARKIS)."""
from __future__ import annotations

import json
import logging
import re

from haftung_ai.config.settings import get_settings
from haftung_ai.llm.client import GroqClient
from haftung_ai.llm.prompts import CONTRADICTION_PROMPT
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)


class ContradictionAgent:
    """Detects semantic contradictions between evidence statements."""

    SEVERITY_PENALTIES = {"direct": 0.3, "partial": 0.2, "tension": 0.1, "none": 0.0}

    def __init__(self):
        settings = get_settings()
        self.llm = GroqClient(temperature=0.0)
        self.enabled = settings.CONTRADICTION_CHECK_ENABLED
        self.max_checks = settings.MAX_CONTRADICTION_CHECKS

    def __call__(self, state: HaftungState) -> HaftungState:
        evidence = state.get("deduplicated_evidence", state.get("evidence", []))

        if len(evidence) < 2 or not self.enabled:
            state["contradictions"] = []
            state["has_contradictions"] = False
            state["contradiction_penalty"] = 0.0
            state["sources_conflict_flag"] = False
            return state

        contradictions = self._detect(evidence)
        penalty = self._calculate_penalty(contradictions)

        state["contradictions"] = contradictions
        state["has_contradictions"] = len(contradictions) > 0
        state["contradiction_penalty"] = penalty
        state["sources_conflict_flag"] = len(contradictions) > 0

        logger.info("ContradictionAgent: %d contradictions, penalty=%.2f", len(contradictions), penalty)
        return state

    def _detect(self, evidence: list) -> list[dict]:
        contradictions = []
        checks = 0
        n = len(evidence)
        for i in range(n):
            if checks >= self.max_checks:
                break
            for j in range(i + 1, n):
                if checks >= self.max_checks:
                    break
                stmt_a = evidence[i].get("statement", "")
                stmt_b = evidence[j].get("statement", "")
                if not stmt_a or not stmt_b:
                    continue
                result = self._check_pair(stmt_a, stmt_b)
                checks += 1
                if result.get("contradiction"):
                    contradictions.append({
                        "statement_a_idx": i,
                        "statement_b_idx": j,
                        "statement_a": stmt_a,
                        "statement_b": stmt_b,
                        "severity": result.get("severity", "none"),
                        "explanation": result.get("explanation", ""),
                    })
        return contradictions

    def _check_pair(self, stmt_a: str, stmt_b: str) -> dict:
        prompt = CONTRADICTION_PROMPT.format(stmt_a=stmt_a, stmt_b=stmt_b)
        try:
            raw = self.llm.invoke(prompt)
            return self._parse(raw)
        except Exception as e:
            logger.error("Contradiction check failed: %s", e)
            return {"contradiction": False, "severity": "none", "explanation": str(e)}

    def _parse(self, text: str) -> dict:
        if not text:
            return {"contradiction": False, "severity": "none", "explanation": ""}
        text = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {"contradiction": False, "severity": "none", "explanation": ""}
        try:
            data = json.loads(text[start : end + 1])
            return {
                "contradiction": bool(data.get("contradiction", False)),
                "severity": data.get("severity", "none"),
                "explanation": data.get("explanation", ""),
            }
        except json.JSONDecodeError:
            return {"contradiction": False, "severity": "none", "explanation": ""}

    def _calculate_penalty(self, contradictions: list[dict]) -> float:
        if not contradictions:
            return 0.0
        total = sum(self.SEVERITY_PENALTIES.get(c.get("severity", "none"), 0.0) for c in contradictions)
        return min(round(total, 2), 0.4)
