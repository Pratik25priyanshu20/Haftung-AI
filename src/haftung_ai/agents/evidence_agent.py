"""Evidence Agent — extracts evidence from RAG chunks (adapted from ARKIS)."""
from __future__ import annotations

import logging

from haftung_ai.llm.client import GroqClient
from haftung_ai.llm.prompts import EVIDENCE_EXTRACTION_PROMPT
from haftung_ai.llm.structured_output import extract_json_array
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)


class EvidenceAgent:
    """Extracts evidence statements from retrieved legal chunks."""

    def __init__(self):
        self.llm = GroqClient(temperature=0.0)

    def __call__(self, state: HaftungState) -> HaftungState:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            state["evidence"] = []
            return state

        formatted_chunks = []
        for c in chunks:
            chunk_id = c.get("chunk_id") or c.get("metadata", {}).get("chunk_id", "unknown")
            formatted_chunks.append(f"[chunk_id={chunk_id}]\n{c.get('content', '')}")

        # Build accident-specific query from state
        query = self._build_query(state)

        prompt = EVIDENCE_EXTRACTION_PROMPT + f"\n\nUnfallfrage:\n{query}\n\nDokumente:\n" + "\n".join(formatted_chunks)
        raw = self.llm.invoke(prompt)
        evidence = extract_json_array(raw)

        state["evidence"] = evidence
        logger.info("EvidenceAgent: extracted %d evidence items", len(evidence))
        return state

    def _build_query(self, state: HaftungState) -> str:
        parts = []
        if state.get("accident_type"):
            parts.append(f"Unfalltyp: {state['accident_type']}")
        if state.get("primary_cause"):
            parts.append(f"Ursache: {state['primary_cause']}")
        summary = state.get("telemetry_summary", {})
        if summary.get("max_speed_kmh"):
            parts.append(f"Geschwindigkeit: {summary['max_speed_kmh']} km/h")
        return " | ".join(parts) if parts else "Unfallanalyse"
