"""Pydantic BaseSettings for Haftung_AI (adapted from ARKIS)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "Haftung_AI"
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    # --- LLM (Groq) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 4096
    GROQ_MAX_RETRIES: int = 3
    GROQ_RATE_LIMIT_RPM: int = 30

    # --- Vector DB (Qdrant) ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "haftung_chunks"

    # --- Embedding ---
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"

    # --- RAG ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    TOP_K_RERANK: int = 20
    HYBRID_ALPHA: float = 0.6
    DEFAULT_RETRIEVAL_STRATEGY: str = "dense"

    # --- Validation ---
    VALIDATION_THRESHOLD: float = 0.7
    GAP_THRESHOLD: float = 0.3
    CONFIDENCE_W_LLM: float = 0.4
    CONFIDENCE_W_COVERAGE: float = 0.3
    CONFIDENCE_W_BASE: float = 0.3

    # --- Evidence & Contradiction ---
    EVIDENCE_CLUSTER_THRESHOLD: float = 0.85
    EVIDENCE_MIN_RELEVANCE: float = 0.3
    CONTRADICTION_CHECK_ENABLED: bool = True
    MAX_CONTRADICTION_CHECKS: int = 15

    # --- Perception ---
    DETECTOR_MODEL: str = "yolov8n"
    TRACKER_TYPE: str = "deepsort"
    CONFIDENCE_THRESHOLD: float = 0.25

    # --- Safety ---
    TTC_WARNING_THRESHOLD: float = 3.0
    TTC_CRITICAL_THRESHOLD: float = 1.5
    MAX_VELOCITY_KMH: float = 200.0

    # --- Paths ---
    DATA_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    KNOWLEDGE_BASE_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "knowledge_base")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
