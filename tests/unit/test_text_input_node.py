"""Tests for TextInputNode and text-mode pipeline components."""
from haftung_ai.agents.orchestrator import TextInputNode
from haftung_ai.types.state import HaftungState


class TestTextInputNode:
    def test_populates_defaults(self):
        node = TextInputNode()
        state: HaftungState = {
            "scenario_text": "Ein Auffahrunfall auf der B27.",
            "variant": "S1",
            "errors": [],
            "warnings": [],
        }
        result = node(state)

        # Vision defaults
        assert result["frames_processed"] == 0
        assert result["tracks"] == []
        assert result["detections_per_frame"] == []
        assert result["scene_graph"] == {}
        assert result["world_models"] == []

        # Telemetry defaults
        assert result["telemetry_summary"] == {}
        assert result["speed_profile"] == {}
        assert result["braking_events"] == []
        assert result["steering_events"] == []
        assert result["ego_states"] == []

    def test_no_overwrite_existing(self):
        """TextInputNode should not overwrite existing state values."""
        node = TextInputNode()
        state: HaftungState = {
            "scenario_text": "Ein Unfall.",
            "variant": "S1",
            "frames_processed": 42,
            "tracks": [{"track_id": 1}],
            "errors": [],
            "warnings": [],
        }
        result = node(state)
        assert result["frames_processed"] == 42
        assert result["tracks"] == [{"track_id": 1}]

    def test_noop_without_scenario_text(self):
        """Without scenario_text, TextInputNode should be a no-op."""
        node = TextInputNode()
        state: HaftungState = {"variant": "S1", "errors": [], "warnings": []}
        result = node(state)
        assert "frames_processed" not in result
        assert "tracks" not in result

    def test_empty_scenario_text_is_noop(self):
        node = TextInputNode()
        state: HaftungState = {
            "scenario_text": "",
            "variant": "S1",
            "errors": [],
            "warnings": [],
        }
        result = node(state)
        assert "frames_processed" not in result
