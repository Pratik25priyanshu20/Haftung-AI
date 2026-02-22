"""Tests for RAGRetrievalNode (mocked retriever)."""
from unittest.mock import MagicMock

import pytest

from haftung_ai.types.state import HaftungState


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = (
        # final_chunks
        [
            {"chunk_id": "stvo_abstand_1", "content": "§4 StVO Abstand", "score": 0.85},
            {"chunk_id": "stvo_geschwindigkeit_1", "content": "§3 StVO", "score": 0.72},
        ],
        # recall_chunks
        [
            {"chunk_id": "stvo_abstand_1", "content": "§4 StVO Abstand", "score": 0.85},
            {"chunk_id": "stvo_geschwindigkeit_1", "content": "§3 StVO", "score": 0.72},
            {"chunk_id": "stvo_vorfahrt_1", "content": "§8 StVO", "score": 0.60},
        ],
        # strategy
        "dense",
    )
    retriever.select_strategy.return_value = "dense"
    return retriever


class TestRAGRetrievalNode:
    def test_text_mode_retrieval(self, mock_retriever):
        from haftung_ai.agents.rag_node import RAGRetrievalNode

        node = RAGRetrievalNode(retriever=mock_retriever)
        state: HaftungState = {
            "scenario_text": "Auffahrunfall auf der B27 wegen zu geringem Abstand.",
            "variant": "S2",
            "errors": [],
            "warnings": [],
        }
        result = node(state)

        assert len(result["retrieved_chunks"]) == 2
        assert len(result["retrieval_recall_chunks"]) == 3
        assert result["retrieval_score"] == 0.85
        assert result["retrieval_latency_s"] > 0
        mock_retriever.retrieve.assert_called_once()

    def test_vision_mode_fallback(self, mock_retriever):
        from haftung_ai.agents.rag_node import RAGRetrievalNode

        node = RAGRetrievalNode(retriever=mock_retriever)
        state: HaftungState = {
            "variant": "S2",
            "tracks": [{"class_name": "car"}, {"class_name": "truck"}],
            "telemetry_summary": {"max_speed_kmh": 80, "emergency_braking": True},
            "impact_frame": 100,
            "errors": [],
            "warnings": [],
        }
        result = node(state)
        assert len(result["retrieved_chunks"]) == 2
        # Verify query was built from vision/telemetry data
        call_args = mock_retriever.retrieve.call_args
        query = call_args[1]["query"] if "query" in call_args[1] else call_args[0][0]
        assert "Geschwindigkeit" in query or "Notbremsung" in query or "Unfall" in query

    def test_empty_state_default_query(self, mock_retriever):
        from haftung_ai.agents.rag_node import RAGRetrievalNode

        node = RAGRetrievalNode(retriever=mock_retriever)
        state: HaftungState = {"variant": "S2", "errors": [], "warnings": []}
        node(state)
        # Should use fallback query
        call_args = mock_retriever.retrieve.call_args
        query = call_args[1].get("query", call_args[0][0] if call_args[0] else "")
        assert "StVO" in query or "Verkehrsunfall" in query

    def test_retrieval_error_handling(self, mock_retriever):
        from haftung_ai.agents.rag_node import RAGRetrievalNode

        mock_retriever.retrieve.side_effect = ConnectionError("Qdrant down")
        node = RAGRetrievalNode(retriever=mock_retriever)
        state: HaftungState = {
            "scenario_text": "Ein Unfall.",
            "variant": "S2",
            "errors": [],
            "warnings": [],
        }
        result = node(state)
        assert result["retrieved_chunks"] == []
        assert result["retrieval_score"] == 0.0
        assert any("RAGRetrievalNode" in e for e in result["errors"])

    def test_query_truncates_long_scenario(self, mock_retriever):
        from haftung_ai.agents.rag_node import RAGRetrievalNode

        node = RAGRetrievalNode(retriever=mock_retriever)
        long_text = "A" * 1000
        state: HaftungState = {
            "scenario_text": long_text,
            "variant": "S2",
            "errors": [],
            "warnings": [],
        }
        # _build_query should truncate to 500 chars
        query = node._build_query(state)
        assert len(query) <= 500
