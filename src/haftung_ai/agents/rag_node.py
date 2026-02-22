"""RAG retrieval node for the LangGraph pipeline."""
from __future__ import annotations

import logging
import time

from haftung_ai.rag.retrieval import AdaptiveRetriever
from haftung_ai.types.state import HaftungState

logger = logging.getLogger(__name__)


class RAGRetrievalNode:
    """LangGraph node that retrieves legal context via RAG.

    Builds a query from scenario_text (text-only mode) or scene/telemetry
    data, calls AdaptiveRetriever, and writes results to state.
    """

    def __init__(self, retriever: AdaptiveRetriever | None = None):
        self._retriever = retriever

    @property
    def retriever(self) -> AdaptiveRetriever:
        if self._retriever is None:
            self._retriever = AdaptiveRetriever()
        return self._retriever

    def _build_query(self, state: HaftungState) -> str:
        """Build retrieval query from available state information."""
        parts: list[str] = []

        # Prefer scenario_text for text-only mode
        scenario_text = state.get("scenario_text", "")
        if scenario_text:
            # Use first 500 chars as query — enough for retrieval, avoids token limits
            parts.append(scenario_text[:500])
        else:
            # Fall back to vision/telemetry summaries
            tracks = state.get("tracks", [])
            if tracks:
                classes = set(t.get("class_name", "") for t in tracks)
                parts.append(f"Unfall mit: {', '.join(c for c in classes if c)}")

            summary = state.get("telemetry_summary", {})
            if summary:
                max_speed = summary.get("max_speed_kmh")
                if max_speed:
                    parts.append(f"Geschwindigkeit: {max_speed} km/h")
                if summary.get("emergency_braking"):
                    parts.append("Notbremsung")

            impact_frame = state.get("impact_frame")
            if impact_frame is not None:
                parts.append("Kollision erkannt")

        if not parts:
            parts.append("Verkehrsunfall Haftung StVO")

        return " ".join(parts)

    def __call__(self, state: HaftungState) -> HaftungState:
        query = self._build_query(state)
        logger.info("RAG query: %s", query[:100])

        start = time.time()
        try:
            final_chunks, recall_chunks, _ = self.retriever.retrieve(
                query=query,
                top_k_recall=20,
                top_k_final=5,
            )
            latency = time.time() - start

            state["retrieved_chunks"] = final_chunks
            state["retrieval_recall_chunks"] = recall_chunks
            state["retrieval_score"] = (
                max(c.get("score", 0.0) for c in final_chunks) if final_chunks else 0.0
            )
            state["retrieval_strategy"] = self.retriever.select_strategy(query)
            state["retrieval_latency_s"] = latency

            logger.info(
                "RAG retrieved %d chunks (recall=%d) in %.2fs",
                len(final_chunks),
                len(recall_chunks),
                latency,
            )
        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            state.setdefault("errors", []).append(f"RAGRetrievalNode: {e}")
            state["retrieved_chunks"] = []
            state["retrieval_recall_chunks"] = []
            state["retrieval_score"] = 0.0
            state["retrieval_latency_s"] = time.time() - start

        return state
