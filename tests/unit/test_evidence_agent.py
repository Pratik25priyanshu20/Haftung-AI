"""Tests for evidence agent (mocked LLM)."""
from unittest.mock import MagicMock, patch

import pytest

from haftung_ai.types.state import HaftungState


@pytest.fixture
def mock_groq():
    with patch("haftung_ai.agents.evidence_agent.GroqClient") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance


def test_evidence_agent_no_chunks(mock_groq):
    from haftung_ai.agents.evidence_agent import EvidenceAgent

    agent = EvidenceAgent()
    state: HaftungState = {
        "retrieved_chunks": [],
        "errors": [],
        "warnings": [],
    }
    result = agent(state)
    assert result["evidence"] == []


def test_evidence_agent_extracts(mock_groq):
    from haftung_ai.agents.evidence_agent import EvidenceAgent

    mock_groq.invoke.return_value = '[{"chunk_id": "stvo_4", "statement": "Abstand halten"}]'
    agent = EvidenceAgent()
    state: HaftungState = {
        "retrieved_chunks": [
            {"chunk_id": "stvo_4", "content": "§ 4 Abs. 1 StVO: Abstand halten"},
        ],
        "primary_cause": "following too closely",
        "accident_type": "rear_end",
        "errors": [],
        "warnings": [],
    }
    result = agent(state)
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["statement"] == "Abstand halten"
