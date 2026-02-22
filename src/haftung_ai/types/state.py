"""HaftungState — the central TypedDict for the LangGraph agent pipeline."""
from __future__ import annotations

from typing import Any, TypedDict


class HaftungState(TypedDict, total=False):
    """Shared state passed through the LangGraph agent pipeline.

    All agents read from and write to this state. Fields are optional
    (total=False) so agents can populate them incrementally.
    """

    # --- Input ---
    video_path: str
    can_log_path: str
    variant: str  # "S1", "S2", "S3"
    scenario_text: str  # text-only mode: German accident narrative

    # --- Vision Agent outputs ---
    frames_processed: int
    detections_per_frame: list[list[dict[str, Any]]]
    tracks: list[dict[str, Any]]
    trajectories: dict[int, list[tuple[float, float]]]
    scene_graph: dict[str, Any]
    impact_frame: int | None
    impact_timestamp: float | None
    world_models: list[dict[str, Any]]

    # --- Telemetry Agent outputs ---
    speed_profile: dict[str, Any]
    braking_events: list[dict[str, Any]]
    steering_events: list[dict[str, Any]]
    ego_states: list[dict[str, Any]]
    telemetry_summary: dict[str, Any]

    # --- RAG (S2/S3 only) ---
    retrieved_chunks: list[dict[str, Any]]
    retrieval_score: float
    retrieval_strategy: str | None
    retrieval_latency_s: float
    retrieval_recall_chunks: list[dict[str, Any]]

    # --- Evidence Agent ---
    evidence: list[dict[str, str]]
    scored_evidence: list[dict[str, Any]]
    deduplicated_evidence: list[dict[str, Any]]

    # --- Contradiction Agent ---
    contradictions: list[dict[str, Any]]
    has_contradictions: bool
    contradiction_penalty: float
    sources_conflict_flag: bool

    # --- Causation Agent ---
    causation_output: dict[str, Any]
    claims: list[dict[str, Any]]
    primary_cause: str
    accident_type: str
    responsibility: list[dict[str, Any]]
    contributing_factors: list[dict[str, Any]]

    # --- Validation Agent ---
    validation_details: dict[str, Any] | None
    confidence_score: float | None
    needs_correction: bool

    # --- Report Agent ---
    report: dict[str, Any]
    report_pdf_path: str | None

    # --- Metadata ---
    errors: list[str]
    warnings: list[str]


def validate_state(state: HaftungState, required_fields: list[str]) -> list[str]:
    """Validate that required fields are present and non-None in state.

    Returns list of missing field names (empty if valid).
    """
    missing = []
    for field_name in required_fields:
        if field_name not in state or state[field_name] is None:  # type: ignore[literal-required]
            missing.append(field_name)
    return missing
