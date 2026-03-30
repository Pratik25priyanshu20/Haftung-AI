"""Tests for embedding service (mocked model)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_st():
    with patch("sentence_transformers.SentenceTransformer") as mock:
        model = MagicMock()
        model.encode.return_value = np.random.randn(3, 768).astype(np.float32)
        model.get_sentence_embedding_dimension.return_value = 768
        mock.return_value = model
        yield model


def test_embed_query(mock_st):
    mock_st.encode.return_value = np.random.randn(768).astype(np.float32)
    from haftung_ai.rag.embeddings import EmbeddingService

    service = EmbeddingService()
    result = service.embed_query("test query")
    assert isinstance(result, list)
    assert len(result) == 768


def test_embed_documents(mock_st):
    mock_st.encode.return_value = np.random.randn(3, 768).astype(np.float32)
    from haftung_ai.rag.embeddings import EmbeddingService

    service = EmbeddingService()
    result = service.embed_documents(["a", "b", "c"])
    assert len(result) == 3


def test_dimension(mock_st):
    from haftung_ai.rag.embeddings import EmbeddingService

    service = EmbeddingService()
    assert service.dimension == 768
