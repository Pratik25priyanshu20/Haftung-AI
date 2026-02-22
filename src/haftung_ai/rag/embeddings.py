"""Embedding service (adapted from ARKIS)."""
from __future__ import annotations

from haftung_ai.config.settings import get_settings


class EmbeddingService:
    """Centralized embedding service using SentenceTransformers."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
