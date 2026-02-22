"""Tests for causation agent (mocked LLM)."""
from unittest.mock import MagicMock, patch

import pytest

from haftung_ai.types.state import HaftungState


@pytest.fixture
def mock_groq():
    with patch("haftung_ai.agents.causation_agent.GroqClient") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance


def _base_state() -> HaftungState:
    return {
        "variant": "S1",
        "scene_graph": {"nodes": {}, "edges": []},
        "tracks": [],
        "speed_profile": {"max_speed_kmh": 50.0, "avg_speed_kmh": 40.0},
        "braking_events": [],
        "steering_events": [],
        "impact_frame": 100,
        "impact_timestamp": 3.33,
        "errors": [],
        "warnings": [],
    }


def test_causation_agent_s1(mock_groq):
    from haftung_ai.agents.causation_agent import CausationAgent

    mock_groq.invoke_json.return_value = {
        "accident_type": "rear_end",
        "primary_cause": "following too closely",
        "contributing_factors": [],
        "responsibility": [{"party": "ego", "percentage": 70.0, "rationale": "too close"}],
        "claims": [],
        "confidence": 0.8,
        "reasoning": "test",
    }
    agent = CausationAgent()
    state = _base_state()
    result = agent(state)
    assert result["primary_cause"] == "following too closely"
    assert result["accident_type"] == "rear_end"


def test_causation_agent_handles_error(mock_groq):
    from haftung_ai.agents.causation_agent import CausationAgent

    mock_groq.invoke_json.side_effect = RuntimeError("API down")
    agent = CausationAgent()
    state = _base_state()
    result = agent(state)
    assert result["primary_cause"] == "Analysis failed"
    assert result["confidence_score"] == 0.0
    assert len(result["errors"]) > 0


def test_causation_agent_text_mode(mock_groq):
    from haftung_ai.agents.causation_agent import CausationAgent

    mock_groq.invoke_json.return_value = {
        "accident_type": "rear_end",
        "primary_cause": "Unzureichender Sicherheitsabstand",
        "contributing_factors": [{"factor": "Ablenkung", "category": "human_error", "severity": "secondary"}],
        "responsibility": [{"party": "Fahrzeug A", "percentage": 100.0, "rationale": "Auffahrender"}],
        "claims": [{"statement": "Der Abstand war zu gering."}],
        "confidence": 0.85,
        "reasoning": "text mode test",
    }
    agent = CausationAgent()
    state = _base_state()
    state["scenario_text"] = "Ein Auffahrunfall auf der B27 wegen zu geringem Sicherheitsabstand."
    result = agent(state)
    assert result["primary_cause"] == "Unzureichender Sicherheitsabstand"
    # In text mode, scenario_text should be used as scene description
    call_args = mock_groq.invoke_json.call_args
    prompt = call_args[0][0]
    assert "Auffahrunfall" in prompt


def test_causation_agent_s2_uses_legal(mock_groq):
    from haftung_ai.agents.causation_agent import CausationAgent

    mock_groq.invoke_json.return_value = {
        "accident_type": "side_collision",
        "primary_cause": "failed to yield",
        "contributing_factors": [],
        "responsibility": [],
        "claims": [],
        "confidence": 0.7,
        "reasoning": "test",
    }
    agent = CausationAgent()
    state = _base_state()
    state["variant"] = "S2"
    state["retrieved_chunks"] = [{"chunk_id": "stvo_4_1", "content": "§ 4 Abs. 1 StVO"}]
    result = agent(state)
    assert result["primary_cause"] == "failed to yield"
