"""Tests for vector store (mocked Qdrant client)."""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_qdrant():
    with patch("qdrant_client.QdrantClient") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


def test_vectorstore_init(mock_qdrant):
    from haftung_ai.rag.vectorstore import VectorStore

    store = VectorStore()
    assert store.client is not None
    assert store.collection_name == "haftung_chunks"


def test_search(mock_qdrant):
    from haftung_ai.rag.vectorstore import VectorStore

    mock_point = MagicMock()
    mock_point.score = 0.95
    mock_point.payload = {"chunk_id": "c1", "content": "test", "metadata": {}}
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_qdrant.query_points.return_value = mock_response

    store = VectorStore()
    results = store.search([0.1] * 768, top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] == 0.95


def test_upsert_length_mismatch(mock_qdrant):
    from haftung_ai.rag.vectorstore import VectorStore

    store = VectorStore()
    with pytest.raises(ValueError, match="mismatch"):
        store.upsert_chunks(
            [{"chunk_id": "1", "content": "x", "metadata": {}}],
            [[0.1] * 768, [0.2] * 768],
        )
