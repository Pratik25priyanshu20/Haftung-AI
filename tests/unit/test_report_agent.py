"""Tests for report agent (mocked LLM)."""
from unittest.mock import MagicMock, patch

import pytest

from haftung_ai.types.state import HaftungState


@pytest.fixture
def mock_groq():
    with patch("haftung_ai.agents.report_agent.GroqClient") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance


def _base_state() -> HaftungState:
    return {
        "variant": "S2",
        "causation_output": {
            "primary_cause": "following too closely",
            "accident_type": "rear_end",
            "responsibility": [{"party": "ego", "percentage": 70.0, "rationale": "too close"}],
            "contributing_factors": [{"category": "distance", "factor": "too close", "severity": "primary"}],
        },
        "primary_cause": "following too closely",
        "accident_type": "rear_end",
        "responsibility": [{"party": "ego", "percentage": 70.0, "rationale": "too close"}],
        "contributing_factors": [{"category": "distance", "factor": "too close", "severity": "primary"}],
        "claims": [],
        "confidence_score": 0.85,
        "frames_processed": 200,
        "impact_frame": 150,
        "telemetry_summary": {"max_speed_kmh": 60.0},
        "errors": [],
        "warnings": [],
    }


def test_report_agent_generates_report(mock_groq):
    from haftung_ai.agents.report_agent import ReportAgent

    mock_groq.invoke_json.return_value = {
        "unfallhergang": "Der Unfall war ein Auffahrunfall.",
        "unfallursache": "Zu geringer Abstand",
        "haftungsverteilung": {"ego": 70, "other": 30},
    }
    agent = ReportAgent()
    state = _base_state()
    result = agent(state)
    assert "report" in result
    assert result["report"]["unfallhergang"] == "Der Unfall war ein Auffahrunfall."


def test_report_agent_no_causation(mock_groq):
    from haftung_ai.agents.report_agent import ReportAgent

    agent = ReportAgent()
    state: HaftungState = {"errors": [], "warnings": []}
    result = agent(state)
    assert len(result["errors"]) > 0


def test_report_agent_handles_error(mock_groq):
    from haftung_ai.agents.report_agent import ReportAgent

    mock_groq.invoke_json.side_effect = RuntimeError("API error")
    agent = ReportAgent()
    state = _base_state()
    result = agent(state)
    assert "report" in result
    assert "fehlgeschlagen" in result["report"]["unfallhergang"]
