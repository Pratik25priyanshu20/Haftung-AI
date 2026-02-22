"""Tests for retrieval quality metrics (pure functions, no mocking needed)."""
import math

from evaluation.metrics.retrieval_quality import (
    _extract_stvo_refs,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    retrieval_quality_metrics,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=5) == 2 / 5

    def test_k_larger_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # Only 2 items but k=5, so 2/5
        assert precision_at_k(retrieved, relevant, k=5) == 2 / 5

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self):
        assert precision_at_k(["a"], set(), k=5) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self):
        assert mean_reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5

    def test_third_is_relevant(self):
        assert abs(mean_reciprocal_rank(["x", "y", "a"], {"a"}) - 1 / 3) < 1e-9

    def test_none_relevant(self):
        assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank([], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert abs(ndcg_at_k(retrieved, relevant, k=3) - 1.0) < 1e-9

    def test_worst_ranking(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_ranking(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4)
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        # IDCG = 1/log2(2) + 1/log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert abs(ndcg_at_k(retrieved, relevant, k=3) - expected) < 1e-9

    def test_empty_inputs(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0
        assert ndcg_at_k(["a"], set(), k=5) == 0.0


class TestExtractStvoRefs:
    def test_chunk_id_mapping(self):
        chunk = {"chunk_id": "stvo_abstand_chunk_1", "content": "some text"}
        refs = _extract_stvo_refs(chunk)
        assert "§4 StVO" in refs

    def test_content_pattern(self):
        chunk = {"chunk_id": "x", "content": "Gemäß §3 StVO darf man..."}
        refs = _extract_stvo_refs(chunk)
        assert "§3 StVO" in refs

    def test_no_refs(self):
        chunk = {"chunk_id": "random", "content": "no legal text here"}
        refs = _extract_stvo_refs(chunk)
        assert len(refs) == 0

    def test_multiple_refs_in_content(self):
        chunk = {"chunk_id": "x", "content": "§4 StVO und §3 StVO sind relevant"}
        refs = _extract_stvo_refs(chunk)
        assert "§4 StVO" in refs
        assert "§3 StVO" in refs


class TestRetrievalQualityMetrics:
    def test_basic_metrics(self):
        predictions = [
            {
                "retrieved_chunks": [
                    {"chunk_id": "stvo_abstand_1", "content": "§4 StVO Abstand"},
                    {"chunk_id": "stvo_geschwindigkeit_1", "content": "§3 StVO"},
                ]
            }
        ]
        ground_truths = [
            {"ground_truth": {"relevant_stvo": ["§4 StVO"]}}
        ]
        result = retrieval_quality_metrics(predictions, ground_truths, k=5)
        assert result["n_evaluated"] == 1
        assert result["precision_at_5"] > 0
        assert result["mrr"] > 0

    def test_no_relevant_stvo_skips(self):
        predictions = [{"retrieved_chunks": [{"chunk_id": "x", "content": "y"}]}]
        ground_truths = [{"ground_truth": {"relevant_stvo": []}}]
        result = retrieval_quality_metrics(predictions, ground_truths, k=5)
        assert result["n_evaluated"] == 0

    def test_empty_inputs(self):
        result = retrieval_quality_metrics([], [], k=5)
        assert result["precision_at_5"] == 0.0
        assert result["mrr"] == 0.0
