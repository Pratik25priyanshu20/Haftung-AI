"""Report Agent — generates accident report."""
from __future__ import annotations

import logging

from haftung_ai.llm.client import GroqClient
from haftung_ai.llm.prompts import REPORT_GENERATION_PROMPT, REPORT_SYSTEM_PROMPT
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)


class ReportAgent:
    """Generates structured accident report from causation analysis."""

    def __init__(self):
        self.llm = GroqClient()

    def __call__(self, state: HaftungState) -> HaftungState:
        causation = state.get("causation_output", {})
        if not causation:
            state.setdefault("errors", []).append("ReportAgent: no causation output")
            return state

        prompt = REPORT_GENERATION_PROMPT.format(
            causation_analysis=self._format_causation(causation),
            scene_description=self._format_scene(state),
            telemetry_summary=self._format_telemetry(state),
        )

        try:
            report = self.llm.invoke_json(prompt, system_prompt=REPORT_SYSTEM_PROMPT)
            state["report"] = report
        except Exception as e:
            logger.error("ReportAgent failed: %s", e)
            state.setdefault("errors", []).append(f"ReportAgent: {e}")
            state["report"] = {
                "accident_sequence": "Report generation failed.",
                "accident_cause": str(causation.get("primary_cause", "Unknown")),
            }

        return state

    def _format_causation(self, causation: dict) -> str:
        parts = [
            f"Accident type: {causation.get('accident_type', 'Unknown')}",
            f"Primary cause: {causation.get('primary_cause', 'Unknown')}",
        ]
        factors = causation.get("contributing_factors", [])
        if factors:
            parts.append("Contributing factors:")
            for f in factors:
                if isinstance(f, dict):
                    parts.append(f"  - {f.get('factor', 'N/A')} ({f.get('severity', '')})")
        responsibility = causation.get("responsibility", [])
        if responsibility:
            parts.append("Liability distribution:")
            for r in responsibility:
                if isinstance(r, dict):
                    parts.append(f"  - {r.get('party', 'N/A')}: {r.get('percentage', 0)}%")
        return "\n".join(parts)

    def _format_scene(self, state: HaftungState) -> str:
        return f"Frames: {state.get('frames_processed', 'N/A')}, Impact: Frame {state.get('impact_frame', 'N/A')}"

    def _format_telemetry(self, state: HaftungState) -> str:
        summary = state.get("telemetry_summary", {})
        return f"Max speed: {summary.get('max_speed_kmh', 'N/A')} km/h"
