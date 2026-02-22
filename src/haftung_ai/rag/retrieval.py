"""Adaptive retrieval pipeline (adapted from ARKIS)."""
from __future__ import annotations

from typing import Any

from haftung_ai.config.settings import get_settings
from haftung_ai.rag.embeddings import EmbeddingService
from haftung_ai.rag.reranker import CrossEncoderReranker
from haftung_ai.rag.vectorstore import VectorStore


class DenseRetriever:
    def __init__(self, vectorstore: VectorStore | None = None, embedder: EmbeddingService | None = None):
        self.vectorstore = vectorstore or VectorStore()
        self.embedder = embedder or EmbeddingService()

    def retrieve(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict[str, Any]]:
        q_emb = self.embedder.embed_query(query)
        return self.vectorstore.search(query_vector=q_emb, top_k=top_k, filters=filters)


class AdaptiveRetriever:
    """Adaptive retriever with dense/hybrid strategy selection."""

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        embedder: EmbeddingService | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        vectorstore = vectorstore or VectorStore()
        embedder = embedder or EmbeddingService()
        self.dense = DenseRetriever(vectorstore, embedder)
        self.reranker = reranker or CrossEncoderReranker()
        settings = get_settings()
        self._default_strategy = settings.DEFAULT_RETRIEVAL_STRATEGY

    def select_strategy(self, query: str) -> str:
        q_lower = query.lower()
        words = query.split()
        if any(w in q_lower for w in ["erkläre", "warum", "wie", "vergleiche"]):
            return "dense"
        if len(words) <= 4:
            return "hybrid"
        if any(word.isupper() and len(word) > 1 for word in words):
            return "hybrid"
        return self._default_strategy

    def retrieve(
        self, query: str, top_k_recall: int = 20, top_k_final: int = 5, filters: dict | None = None
    ) -> tuple[list[dict], list[dict], list]:
        recall_chunks = self.dense.retrieve(query, top_k=top_k_recall, filters=filters)
        final_chunks, rerank_all = self.reranker.rerank(query, recall_chunks, top_k=top_k_final)
        return final_chunks, recall_chunks, rerank_all
