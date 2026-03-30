"""Tests for all evaluation metrics."""
from __future__ import annotations

from evaluation.metrics.aggregate import aggregate_metrics
from evaluation.metrics.calibration import brier_score, expected_calibration_error
from evaluation.metrics.causation_accuracy import causation_accuracy
from evaluation.metrics.factors_f1 import factors_f1
from evaluation.metrics.hallucination import hallucination_by_source, hallucination_rate
from evaluation.metrics.hallucination_entropy import answer_entropy, cause_stability
from evaluation.metrics.responsibility_mae import responsibility_mae

# --- Causation Accuracy ---

def test_causation_accuracy_perfect():
    preds = [{"primary_cause": "speeding"}, {"primary_cause": "tailgating"}]
    gts = [{"primary_cause": "speeding"}, {"primary_cause": "tailgating"}]
    assert causation_accuracy(preds, gts) == 1.0


def test_causation_accuracy_none():
    preds = [{"primary_cause": "speeding"}, {"primary_cause": "distraction"}]
    gts = [{"primary_cause": "tailgating"}, {"primary_cause": "fatigue"}]
    assert causation_accuracy(preds, gts) == 0.0


def test_causation_accuracy_partial():
    preds = [{"primary_cause": "speeding"}, {"primary_cause": "wrong"}]
    gts = [{"primary_cause": "speeding"}, {"primary_cause": "tailgating"}]
    assert causation_accuracy(preds, gts) == 0.5


def test_causation_accuracy_case_insensitive():
    preds = [{"primary_cause": "SPEEDING"}]
    gts = [{"primary_cause": "speeding"}]
    assert causation_accuracy(preds, gts) == 1.0


def test_causation_accuracy_empty():
    assert causation_accuracy([], []) == 0.0


# --- Responsibility MAE ---

def test_responsibility_mae_perfect():
    preds = [{"responsibility": [{"party": "ego", "percentage": 70.0}]}]
    gts = [{"responsibility": [{"party": "ego", "percentage": 70.0}]}]
    assert responsibility_mae(preds, gts) == 0.0


def test_responsibility_mae_off():
    preds = [{"responsibility": [{"party": "ego", "percentage": 80.0}]}]
    gts = [{"responsibility": [{"party": "ego", "percentage": 70.0}]}]
    assert responsibility_mae(preds, gts) == 10.0


def test_responsibility_mae_empty():
    assert responsibility_mae([], []) == 0.0


# --- Factors F1 ---

def test_factors_f1_perfect():
    preds = [{"contributing_factors": [{"category": "speed"}, {"category": "distance"}]}]
    gts = [{"contributing_factors": [{"category": "speed"}, {"category": "distance"}]}]
    result = factors_f1(preds, gts)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_factors_f1_no_match():
    preds = [{"contributing_factors": [{"category": "speed"}]}]
    gts = [{"contributing_factors": [{"category": "visibility"}]}]
    result = factors_f1(preds, gts)
    assert result["f1"] == 0.0


def test_factors_f1_empty():
    result = factors_f1([], [])
    assert result["f1"] == 0.0


# --- Hallucination Rate ---

def test_hallucination_rate_all_supported():
    preds = [{"claims": [{"supported": True}, {"supported": True}]}]
    assert hallucination_rate(preds) == 0.0


def test_hallucination_rate_all_unsupported():
    preds = [{"claims": [{"supported": False}, {"supported": False}]}]
    assert hallucination_rate(preds) == 1.0


def test_hallucination_rate_mixed():
    preds = [{"claims": [{"supported": True}, {"supported": False}]}]
    assert hallucination_rate(preds) == 0.5


def test_hallucination_rate_no_claims():
    preds = [{"claims": []}]
    assert hallucination_rate(preds) == 0.0


def test_hallucination_by_source():
    preds = [
        {"claims": [
            {"supported": True, "source_type": "vision"},
            {"supported": False, "source_type": "rag"},
            {"supported": False, "source_type": "rag"},
        ]}
    ]
    result = hallucination_by_source(preds)
    assert result["vision"] == 0.0
    assert result["rag"] == 1.0


# --- Entropy / Stability ---

def test_answer_entropy_consistent():
    assert answer_entropy(["a", "a", "a"]) == 0.0


def test_answer_entropy_mixed():
    e = answer_entropy(["a", "b"])
    assert e > 0.9  # log2(2) = 1.0


def test_answer_entropy_empty():
    assert answer_entropy([]) == 0.0


def test_cause_stability_consistent():
    reruns = [
        [{"primary_cause": "speeding"}, {"primary_cause": "speeding"}, {"primary_cause": "speeding"}],
    ]
    result = cause_stability(reruns)
    assert result["avg_entropy"] == 0.0
    assert result["consistency_rate"] == 1.0


def test_cause_stability_inconsistent():
    reruns = [
        [{"primary_cause": "speeding"}, {"primary_cause": "tailgating"}, {"primary_cause": "distraction"}],
    ]
    result = cause_stability(reruns)
    assert result["avg_entropy"] > 0
    assert result["consistency_rate"] == 0.0


# --- Calibration ---

def test_ece_perfect():
    ece = expected_calibration_error([1.0, 1.0], [True, True])
    assert ece < 0.1


def test_ece_overconfident():
    ece = expected_calibration_error([0.9, 0.9, 0.9, 0.9], [True, False, True, False])
    assert ece > 0.3


def test_ece_empty():
    assert expected_calibration_error([], []) == 0.0


def test_brier_score_perfect():
    bs = brier_score([1.0, 0.0], [True, False])
    assert bs == 0.0


def test_brier_score_worst():
    bs = brier_score([0.0, 1.0], [True, False])
    assert bs == 1.0


def test_brier_score_empty():
    assert brier_score([], []) == 0.0


# --- Aggregate ---

def test_aggregate_basic():
    preds = [
        {
            "primary_cause": "speeding",
            "responsibility": [{"party": "ego", "percentage": 70.0}],
            "contributing_factors": [{"category": "speed"}],
            "claims": [{"supported": True}],
            "confidence": 0.8,
        }
    ]
    gts = [
        {
            "primary_cause": "speeding",
            "responsibility": [{"party": "ego", "percentage": 70.0}],
            "contributing_factors": [{"category": "speed"}],
        }
    ]
    result = aggregate_metrics(preds, gts)
    assert "causation_accuracy_fuzzy" in result
    assert "causation_accuracy_taxonomy" in result
    assert "factors_f1" in result
    assert "hallucination_rate" in result
    assert "ece" in result
    assert "brier_score" in result
    assert "_metric_notes" in result
    assert "_synthetic_data_notice" in result


def test_aggregate_with_reruns():
    preds = [{"primary_cause": "speeding", "confidence": 0.8, "claims": []}]
    gts = [{"primary_cause": "speeding"}]
    reruns = [[{"primary_cause": "speeding"}, {"primary_cause": "speeding"}]]
    result = aggregate_metrics(preds, gts, rerun_results=reruns)
    assert "avg_entropy" in result
    assert "consistency_rate" in result
