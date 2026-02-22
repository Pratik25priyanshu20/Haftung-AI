"""LangGraph orchestrator — builds 3-variant graphs at compile time."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from haftung_ai.agents.causation_agent import CausationAgent
from haftung_ai.agents.contradiction_agent import ContradictionAgent
from haftung_ai.agents.evidence_agent import EvidenceAgent
from haftung_ai.agents.rag_node import RAGRetrievalNode
from haftung_ai.agents.report_agent import ReportAgent
from haftung_ai.agents.telemetry_agent import TelemetryAgent
from haftung_ai.agents.validation_agent import ValidationAgent
from haftung_ai.agents.vision_agent import VisionAgent
from haftung_ai.types.state import HaftungState


class TextInputNode:
    """Populates empty vision/telemetry fields from scenario_text for text-only mode.

    In text-only mode there is no video or CAN log, so this node fills
    placeholder values so downstream agents don't break.
    """

    def __call__(self, state: HaftungState) -> HaftungState:
        scenario_text = state.get("scenario_text", "")
        if not scenario_text:
            return state

        # Provide sensible defaults for vision fields
        if "frames_processed" not in state:
            state["frames_processed"] = 0
        if "tracks" not in state:
            state["tracks"] = []
        if "detections_per_frame" not in state:
            state["detections_per_frame"] = []
        if "scene_graph" not in state:
            state["scene_graph"] = {}
        if "world_models" not in state:
            state["world_models"] = []

        # Provide sensible defaults for telemetry fields
        if "telemetry_summary" not in state:
            state["telemetry_summary"] = {}
        if "speed_profile" not in state:
            state["speed_profile"] = {}
        if "braking_events" not in state:
            state["braking_events"] = []
        if "steering_events" not in state:
            state["steering_events"] = []
        if "ego_states" not in state:
            state["ego_states"] = []

        return state


def build_graph(variant: str = "S1"):
    """Build a structurally different graph per variant.

    S1: Vision -> Telemetry -> Causation -> Report
    S2: Vision -> Telemetry -> RAG -> Evidence -> Contradiction -> Causation -> Validation -> Report
    S3: S2 + claim-level constraint enforcement in validation
    """
    graph = StateGraph(HaftungState)

    # Common nodes
    graph.add_node("vision", VisionAgent())
    graph.add_node("telemetry", TelemetryAgent())
    graph.add_node("causation", CausationAgent())
    graph.add_node("report", ReportAgent())

    if variant == "S1":
        # No RAG, no validation
        graph.set_entry_point("vision")
        graph.add_edge("vision", "telemetry")
        graph.add_edge("telemetry", "causation")
        graph.add_edge("causation", "report")
        graph.add_edge("report", END)

    elif variant in ("S2", "S3"):
        # Add RAG + evidence + contradiction + validation nodes
        graph.add_node("rag_retrieval", RAGRetrievalNode())
        graph.add_node("evidence", EvidenceAgent())
        graph.add_node("contradiction", ContradictionAgent())
        graph.add_node("validation", ValidationAgent())

        graph.set_entry_point("vision")
        graph.add_edge("vision", "telemetry")
        graph.add_edge("telemetry", "rag_retrieval")
        graph.add_edge("rag_retrieval", "causation")
        graph.add_edge("causation", "evidence")
        graph.add_edge("evidence", "contradiction")
        graph.add_edge("contradiction", "validation")

        if variant == "S3":
            # S3: validation can loop back to causation if needs_correction
            graph.add_conditional_edges(
                "validation",
                lambda s: "causation" if s.get("needs_correction") else "report",
                {"causation": "causation", "report": "report"},
            )
        else:
            graph.add_edge("validation", "report")

        graph.add_edge("report", END)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return graph.compile()


def build_text_graph(variant: str = "S1"):
    """Build a text-only graph for scenario-based evaluation.

    S1: TextInput -> Causation -> Report
    S2: TextInput -> RAG -> Causation -> Evidence -> Contradiction -> Validation -> Report
    S3: S2 + claim-level constraint enforcement in validation
    """
    graph = StateGraph(HaftungState)

    # Common nodes for text mode
    graph.add_node("text_input", TextInputNode())
    graph.add_node("causation", CausationAgent())
    graph.add_node("report", ReportAgent())

    if variant == "S1":
        graph.set_entry_point("text_input")
        graph.add_edge("text_input", "causation")
        graph.add_edge("causation", "report")
        graph.add_edge("report", END)

    elif variant in ("S2", "S3"):
        graph.add_node("rag_retrieval", RAGRetrievalNode())
        graph.add_node("evidence", EvidenceAgent())
        graph.add_node("contradiction", ContradictionAgent())
        graph.add_node("validation", ValidationAgent())

        graph.set_entry_point("text_input")
        graph.add_edge("text_input", "rag_retrieval")
        graph.add_edge("rag_retrieval", "causation")
        graph.add_edge("causation", "evidence")
        graph.add_edge("evidence", "contradiction")
        graph.add_edge("contradiction", "validation")

        if variant == "S3":
            graph.add_conditional_edges(
                "validation",
                lambda s: "causation" if s.get("needs_correction") else "report",
                {"causation": "causation", "report": "report"},
            )
        else:
            graph.add_edge("validation", "report")

        graph.add_edge("report", END)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return graph.compile()


def run_analysis(video_path: str, can_log_path: str, variant: str = "S2") -> HaftungState:
    """Convenience function to run full analysis pipeline."""
    graph = build_graph(variant)
    initial_state: HaftungState = {
        "video_path": video_path,
        "can_log_path": can_log_path,
        "variant": variant,
        "errors": [],
        "warnings": [],
    }
    return graph.invoke(initial_state)


def run_text_analysis(scenario_text: str, variant: str = "S2") -> HaftungState:
    """Convenience function to run text-only analysis pipeline."""
    graph = build_text_graph(variant)
    initial_state: HaftungState = {
        "scenario_text": scenario_text,
        "variant": variant,
        "errors": [],
        "warnings": [],
    }
    return graph.invoke(initial_state)
