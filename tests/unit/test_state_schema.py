"""Tests for HaftungState and validate_state."""
from __future__ import annotations

import pytest

from haftung_ai.types.causation import CausationOutput, Claim, ContributingFactor, ResponsibilityAssignment
from haftung_ai.types.state import HaftungState, validate_state


class TestHaftungState:
    def test_empty_state(self):
        state: HaftungState = {}
        assert isinstance(state, dict)

    def test_state_with_video_path(self):
        state: HaftungState = {"video_path": "/path/to/video.mp4", "variant": "S1"}
        assert state["video_path"] == "/path/to/video.mp4"
        assert state["variant"] == "S1"

    def test_state_all_fields(self):
        state: HaftungState = {
            "video_path": "video.mp4",
            "can_log_path": "can.csv",
            "variant": "S2",
            "frames_processed": 100,
            "tracks": [],
            "impact_frame": 50,
            "primary_cause": "following too closely",
            "errors": [],
            "warnings": [],
        }
        assert state["frames_processed"] == 100
        assert state["impact_frame"] == 50

    def test_state_incremental_population(self):
        state: HaftungState = {"variant": "S3"}
        state["video_path"] = "test.mp4"
        state["tracks"] = [{"track_id": 1}]
        state["causation_output"] = {"primary_cause": "speeding"}
        assert len(state["tracks"]) == 1


class TestValidateState:
    def test_all_fields_present(self):
        state: HaftungState = {"video_path": "v.mp4", "variant": "S1"}
        missing = validate_state(state, ["video_path", "variant"])
        assert missing == []

    def test_missing_fields(self):
        state: HaftungState = {"video_path": "v.mp4"}
        missing = validate_state(state, ["video_path", "variant", "tracks"])
        assert "variant" in missing
        assert "tracks" in missing
        assert "video_path" not in missing

    def test_none_value_counts_as_missing(self):
        state: HaftungState = {"video_path": "v.mp4", "impact_frame": None}
        missing = validate_state(state, ["video_path", "impact_frame"])
        assert "impact_frame" in missing

    def test_empty_required_fields(self):
        state: HaftungState = {"video_path": "v.mp4"}
        missing = validate_state(state, [])
        assert missing == []

    def test_all_missing(self):
        state: HaftungState = {}
        missing = validate_state(state, ["video_path", "variant"])
        assert len(missing) == 2


class TestCausationOutput:
    def test_basic_causation(self):
        output = CausationOutput(
            accident_type="rear_end",
            primary_cause="Following too closely",
            confidence=0.85,
            variant="S2",
        )
        assert output.accident_type == "rear_end"
        assert output.confidence == 0.85

    def test_full_causation(self):
        output = CausationOutput(
            accident_type="intersection",
            primary_cause="Running red light",
            contributing_factors=[
                ContributingFactor(
                    factor="Excessive speed",
                    category="speed",
                    severity="secondary",
                    legal_reference="§ 3 StVO",
                )
            ],
            responsibility=[
                ResponsibilityAssignment(party="ego", percentage=30.0, rationale="Contributory negligence"),
                ResponsibilityAssignment(party="other_1", percentage=70.0, rationale="Red light violation"),
            ],
            claims=[
                Claim(statement="Ego vehicle was traveling at 55 km/h", source_type="telemetry", confidence=0.9),
                Claim(statement="Other vehicle entered intersection on red", source_type="vision", confidence=0.8),
            ],
            legal_references=["§ 37 StVO", "BGH VI ZR 123/20"],
            confidence=0.78,
            variant="S3",
        )
        assert len(output.contributing_factors) == 1
        assert len(output.responsibility) == 2
        assert sum(r.percentage for r in output.responsibility) == 100.0
        assert len(output.claims) == 2

    def test_claim_validation_fields(self):
        claim = Claim(
            statement="Speed was 80 km/h",
            source_type="telemetry",
            source_id="can_t=1.5",
            confidence=0.95,
            supported=True,
        )
        assert claim.supported is True
        assert claim.source_type == "telemetry"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            CausationOutput(
                accident_type="test",
                primary_cause="test",
                confidence=1.5,
                variant="S1",
            )
