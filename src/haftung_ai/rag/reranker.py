"""Cross-encoder reranker (from ARKIS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RerankResult:
    chunk: dict[str, Any]
    rerank_score: float


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-large."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str | None = None, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        import torch

        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                inputs = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits.squeeze(-1)
                batch_scores = logits.detach().float().cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        return scores

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> tuple[list[dict[str, Any]], list[RerankResult]]:
        if not chunks:
            return [], []
        passages = [c["content"] for c in chunks]
        scores = self.score(query, passages)
        results = [RerankResult(chunk=c, rerank_score=s) for c, s in zip(chunks, scores)]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        top = results[:top_k]
        return [{**r.chunk, "rerank_score": r.rerank_score} for r in top], results
