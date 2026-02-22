"""Causation Agent — LLM reasoning with S1/S2/S3 modes."""
from __future__ import annotations

import logging

from haftung_ai.llm.client import GroqClient
from haftung_ai.llm.prompts import (
    CAUSATION_S1_PROMPT,
    CAUSATION_S2_PROMPT,
    CAUSATION_S3_PROMPT,
    CAUSATION_SYSTEM_PROMPT,
)
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)


class CausationAgent:
    """LLM-based accident causation analysis with 3 system variants."""

    def __init__(self):
        self.llm = GroqClient()

    def __call__(self, state: HaftungState) -> HaftungState:
        variant = state.get("variant", "S1")

        scene_desc = self._build_scene_description(state)
        telemetry_summary = self._build_telemetry_summary(state)
        impact_details = self._build_impact_details(state)

        if variant == "S1":
            prompt = CAUSATION_S1_PROMPT.format(
                scene_description=scene_desc,
                telemetry_summary=telemetry_summary,
                impact_details=impact_details,
            )
        elif variant == "S2":
            legal_context = self._build_legal_context(state)
            prompt = CAUSATION_S2_PROMPT.format(
                scene_description=scene_desc,
                telemetry_summary=telemetry_summary,
                impact_details=impact_details,
                legal_context=legal_context,
            )
        else:  # S3
            legal_context = self._build_legal_context(state)
            evidence_summary = self._build_evidence_summary(state)
            prompt = CAUSATION_S3_PROMPT.format(
                scene_description=scene_desc,
                telemetry_summary=telemetry_summary,
                impact_details=impact_details,
                legal_context=legal_context,
                evidence_summary=evidence_summary,
            )

        try:
            raw = self.llm.invoke_json(prompt, system_prompt=CAUSATION_SYSTEM_PROMPT)
            # Normalize into CausationOutput fields
            state["causation_output"] = raw
            state["primary_cause"] = raw.get("primary_cause", "Unknown")
            state["accident_type"] = raw.get("accident_type", "unknown")
            state["responsibility"] = raw.get("responsibility", [])
            state["contributing_factors"] = raw.get("contributing_factors", [])
            state["claims"] = raw.get("claims", [])
            state["confidence_score"] = raw.get("confidence", 0.5)
        except Exception as e:
            logger.error("CausationAgent failed: %s", e)
            state.setdefault("errors", []).append(f"CausationAgent: {e}")
            state["primary_cause"] = "Analysis failed"
            state["confidence_score"] = 0.0

        return state

    def _build_scene_description(self, state: HaftungState) -> str:
        # Text-only mode: use scenario_text directly as scene description
        scenario_text = state.get("scenario_text", "")
        if scenario_text:
            return scenario_text

        tracks = state.get("tracks", [])
        frames = state.get("frames_processed", 0)
        if not tracks:
            return f"Video mit {frames} Frames verarbeitet. Keine Objekte erkannt."
        unique_tracks = len(set(t.get("track_id") for t in tracks))
        classes = set(t.get("class_name") for t in tracks)
        return f"{frames} Frames, {unique_tracks} verfolgte Objekte ({', '.join(classes)})"

    def _build_telemetry_summary(self, state: HaftungState) -> str:
        summary = state.get("telemetry_summary", {})
        if not summary:
            return "Keine Telemetriedaten verfügbar."
        parts = [f"Max. Geschwindigkeit: {summary.get('max_speed_kmh', 'N/A')} km/h"]
        if summary.get("emergency_braking"):
            parts.append("Notbremsung erkannt")
        parts.append(f"Bremsmanöver: {summary.get('num_braking_events', 0)}")
        return "\n".join(parts)

    def _build_impact_details(self, state: HaftungState) -> str:
        frame = state.get("impact_frame")
        ts = state.get("impact_timestamp")
        if frame is None:
            return "Kein Aufprall detektiert."
        return f"Aufprall bei Frame {frame} (t={ts:.2f}s)" if ts else f"Aufprall bei Frame {frame}"

    def _build_legal_context(self, state: HaftungState) -> str:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return "Keine rechtlichen Referenzen verfügbar."
        return "\n\n".join(f"[{c.get('chunk_id', 'N/A')}]: {c.get('content', '')}" for c in chunks[:5])

    def _build_evidence_summary(self, state: HaftungState) -> str:
        evidence = state.get("deduplicated_evidence", state.get("evidence", []))
        if not evidence:
            return "Keine Beweismittel verfügbar."
        return "\n".join(f"- {e.get('statement', '')} (Quelle: {e.get('chunk_id', 'N/A')})" for e in evidence)
