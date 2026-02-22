"""Qdrant vector store (adapted from ARKIS)."""
from __future__ import annotations

from typing import Any

from haftung_ai.config.settings import get_settings


class VectorStore:
    """Qdrant wrapper for Haftung_AI knowledge base."""

    def __init__(self):
        from qdrant_client import QdrantClient

        settings = get_settings()
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.QDRANT_COLLECTION

    def ensure_collection(self, vector_dim: int) -> None:
        from qdrant_client.http import models as qmodels

        existing = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in existing):
            return
        self.client.create_collection(
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
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_vector: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from qdrant_client.http import models as qmodels

        qfilter = None
        if filters:
            conditions = [qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)) for k, v in filters.items()]
            qfilter = qmodels.Filter(must=conditions)
        response = self.client.query_points(
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
