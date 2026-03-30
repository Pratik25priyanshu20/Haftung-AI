"""Tests for Groq client (mocked, no actual API calls)."""
from unittest.mock import MagicMock, patch

import pytest

from haftung_ai.llm.client import GroqClient


@pytest.fixture
def mock_settings():
    with patch("haftung_ai.llm.client.get_settings") as mock:
        settings = MagicMock()
        settings.GROQ_API_KEY = "test-key"
        settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        settings.GROQ_TEMPERATURE = 0.1
        settings.GROQ_MAX_TOKENS = 4096
        settings.GROQ_MAX_RETRIES = 1
        settings.GROQ_RATE_LIMIT_RPM = 0
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_groq_client(mock_settings):
    with patch("groq.Groq") as mock_groq_cls:
        groq_instance = MagicMock()
        mock_groq_cls.return_value = groq_instance
        client = GroqClient()
        yield client, groq_instance


def test_client_init(mock_settings):
    client = GroqClient()
    assert client.model == "llama-3.3-70b-versatile"
    assert client.api_key == "test-key"


def test_client_model_override(mock_settings):
    client = GroqClient(model="custom-model")
    assert client.model == "custom-model"


def test_invoke(mock_groq_client):
    client, groq = mock_groq_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
    groq.chat.completions.create.return_value = mock_response

    result = client.invoke("test prompt")
    assert result == "test response"
    groq.chat.completions.create.assert_called_once()


def test_invoke_with_system(mock_groq_client):
    client, groq = mock_groq_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="response"))]
    groq.chat.completions.create.return_value = mock_response

    client.invoke("user msg", system_prompt="system msg")
    call_args = groq.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_invoke_json(mock_groq_client):
    client, groq = mock_groq_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]
    groq.chat.completions.create.return_value = mock_response

    result = client.invoke_json("test prompt")
    assert result == {"key": "value"}


def test_judge_zero_temp(mock_groq_client):
    client, groq = mock_groq_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="8/10"))]
    groq.chat.completions.create.return_value = mock_response

    result = client.judge("rate this")
    assert result == "8/10"
    call_args = groq.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 0.0
