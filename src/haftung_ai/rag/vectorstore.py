"""Qdrant vector store (adapted from ARKIS)."""
from __future__ import annotations

import logging
from typing import Any

from haftung_ai.config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant wrapper for Haftung_AI knowledge base."""

    def __init__(self):
        settings = get_settings()
        self._url = settings.QDRANT_URL
        self.collection_name = settings.QDRANT_COLLECTION

    def _make_client(self) -> Any:
        from qdrant_client import QdrantClient

        return QdrantClient(url=self._url, timeout=10)

    @property
    def client(self) -> Any:
        return self._make_client()

    def ensure_collection(self, vector_dim: int) -> None:
        from qdrant_client.http import models as qmodels

        c = self._make_client()
        existing = c.get_collections().collections
        if any(col.name == self.collection_name for col in existing):
            return
        c.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=vector_dim, distance=qmodels.Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        from qdrant_client.http import models as qmodels

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        points = []
        for chunk, vec in zip(chunks, embeddings):
            points.append(
                qmodels.PointStruct(
                    id=chunk["chunk_id"],
                    vector=vec,
                    payload={"chunk_id": chunk["chunk_id"], "content": chunk["content"], "metadata": chunk["metadata"]},
                )
            )
        c = self._make_client()
        c.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_vector: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from qdrant_client.http import models as qmodels

        qfilter = None
        if filters:
            conditions = [qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)) for k, v in filters.items()]
            qfilter = qmodels.Filter(must=conditions)

        c = self._make_client()
        response = c.query_points(
            collection_name=self.collection_name, query=query_vector, limit=top_k, query_filter=qfilter, with_payload=True
        )
        return [
            {
                "score": float(r.score),
                "chunk_id": (r.payload or {}).get("chunk_id"),
                "content": (r.payload or {}).get("content"),
                "metadata": (r.payload or {}).get("metadata", {}),
            }
            for r in response.points
        ]
