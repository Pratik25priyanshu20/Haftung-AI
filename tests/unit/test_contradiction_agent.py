"""Tests for contradiction agent (mocked LLM)."""
from unittest.mock import MagicMock, patch

import pytest

from haftung_ai.types.state import HaftungState


@pytest.fixture
def mock_deps():
    with (
        patch("haftung_ai.agents.contradiction_agent.GroqClient") as mock_groq,
        patch("haftung_ai.agents.contradiction_agent.get_settings") as mock_settings,
    ):
        groq = MagicMock()
        settings = MagicMock()
        settings.CONTRADICTION_CHECK_ENABLED = True
        settings.MAX_CONTRADICTION_CHECKS = 15
        mock_groq.return_value = groq
        mock_settings.return_value = settings
        yield groq, settings


def test_no_contradictions_few_evidence(mock_deps):
    from haftung_ai.agents.contradiction_agent import ContradictionAgent

    agent = ContradictionAgent()
    state: HaftungState = {
        "evidence": [{"statement": "Single item"}],
        "errors": [],
        "warnings": [],
    }
    result = agent(state)
    assert result["has_contradictions"] is False
    assert result["contradiction_penalty"] == 0.0


def test_no_contradictions_detected(mock_deps):
    from haftung_ai.agents.contradiction_agent import ContradictionAgent

    groq, _ = mock_deps
    groq.invoke.return_value = '{"contradiction": false, "severity": "none", "explanation": ""}'
    agent = ContradictionAgent()
    state: HaftungState = {
        "evidence": [
            {"statement": "Speed was 50 km/h"},
            {"statement": "Driver was traveling at normal speed"},
        ],
        "errors": [],
        "warnings": [],
    }
    result = agent(state)
    assert result["has_contradictions"] is False


def test_contradiction_detected(mock_deps):
    from haftung_ai.agents.contradiction_agent import ContradictionAgent

    groq, _ = mock_deps
    groq.invoke.return_value = '{"contradiction": true, "severity": "direct", "explanation": "Speed mismatch"}'
    agent = ContradictionAgent()
    state: HaftungState = {
        "evidence": [
            {"statement": "Speed was 50 km/h"},
            {"statement": "Speed was 80 km/h"},
        ],
        "errors": [],
        "warnings": [],
    }
    result = agent(state)
    assert result["has_contradictions"] is True
    assert len(result["contradictions"]) == 1
    assert result["contradiction_penalty"] > 0


def test_penalty_capped(mock_deps):
    from haftung_ai.agents.contradiction_agent import ContradictionAgent

    groq, _ = mock_deps
    groq.invoke.return_value = '{"contradiction": true, "severity": "direct", "explanation": "conflict"}'
    agent = ContradictionAgent()
    # Create many evidence items to trigger multiple contradictions
    state: HaftungState = {
        "evidence": [{"statement": f"Statement {i}"} for i in range(5)],
        "errors": [],
        "warnings": [],
    }
    result = agent(state)
    assert result["contradiction_penalty"] <= 0.4
