"""Tests for settings configuration."""
from __future__ import annotations

from haftung_ai.config.settings import Settings, get_settings


class TestSettings:
    def test_default_values(self):
        settings = Settings()
        assert settings.APP_NAME == "Haftung_AI"
        assert settings.GROQ_MODEL == "llama-3.3-70b-versatile"
        assert settings.QDRANT_URL == "http://localhost:6333"
        assert settings.QDRANT_COLLECTION == "haftung_chunks"
        assert settings.EMBEDDING_MODEL == "BAAI/bge-large-en-v1.5"

    def test_groq_settings(self):
        settings = Settings()
        assert settings.GROQ_TEMPERATURE == 0.1
        assert settings.GROQ_MAX_TOKENS == 4096
        assert settings.GROQ_MAX_RETRIES == 3
        assert settings.GROQ_RATE_LIMIT_RPM == 30

    def test_rag_settings(self):
        settings = Settings()
        assert settings.CHUNK_SIZE == 1000
        assert settings.CHUNK_OVERLAP == 200
        assert settings.TOP_K_RETRIEVAL == 5
        assert settings.TOP_K_RERANK == 20
        assert settings.HYBRID_ALPHA == 0.6

    def test_validation_settings(self):
        settings = Settings()
        assert settings.VALIDATION_THRESHOLD == 0.7
        assert settings.GAP_THRESHOLD == 0.3

    def test_evidence_settings(self):
        settings = Settings()
        assert settings.EVIDENCE_CLUSTER_THRESHOLD == 0.85
        assert settings.EVIDENCE_MIN_RELEVANCE == 0.3
        assert settings.CONTRADICTION_CHECK_ENABLED is True
        assert settings.MAX_CONTRADICTION_CHECKS == 15

    def test_perception_settings(self):
        settings = Settings()
        assert settings.DETECTOR_MODEL == "yolov8n"
        assert settings.TRACKER_TYPE == "deepsort"
        assert settings.CONFIDENCE_THRESHOLD == 0.25

    def test_safety_settings(self):
        settings = Settings()
        assert settings.TTC_WARNING_THRESHOLD == 3.0
        assert settings.TTC_CRITICAL_THRESHOLD == 1.5
        assert settings.MAX_VELOCITY_KMH == 200.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
        settings = Settings()
        assert settings.GROQ_MODEL == "llama-3.1-8b-instant"
        assert settings.QDRANT_COLLECTION == "test_collection"

    def test_get_settings_returns_settings(self):
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.APP_NAME == "Haftung_AI"

    def test_paths(self):
        settings = Settings()
        assert settings.DATA_DIR.name == "data"
        assert settings.KNOWLEDGE_BASE_DIR.name == "knowledge_base"
