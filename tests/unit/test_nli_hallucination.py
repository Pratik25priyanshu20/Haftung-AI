"""Tests for NLI hallucination checker (mocked model)."""
from unittest.mock import MagicMock

import pytest

from evaluation.metrics.nli_hallucination import (
    NLIHallucinationChecker,
    nli_hallucination_rate,
)


@pytest.fixture
def mock_checker():
    """Create a mock NLI checker that returns controllable results."""
    checker = MagicMock(spec=NLIHallucinationChecker)
    return checker


class TestNLIHallucinationRate:
    def test_all_entailed(self, mock_checker):
        mock_checker.check_claim.return_value = {
            "label": "entailment",
            "scores": {"contradiction": 0.05, "entailment": 0.90, "neutral": 0.05},
            "hallucinated": False,
        }
        predictions = [
            {"claims": [{"statement": "claim 1"}, {"statement": "claim 2"}]}
        ]
        scenarios = [{"scenario_text": "Some premise text."}]

        result = nli_hallucination_rate(predictions, scenarios, checker=mock_checker)
        assert result["nli_entailment_rate"] == 1.0
        assert result["nli_contradiction_rate"] == 0.0
        assert result["nli_hallucination_rate"] == 0.0
        assert result["total_claims_checked"] == 2

    def test_all_contradictions(self, mock_checker):
        mock_checker.check_claim.return_value = {
            "label": "contradiction",
            "scores": {"contradiction": 0.90, "entailment": 0.05, "neutral": 0.05},
            "hallucinated": True,
        }
        predictions = [{"claims": [{"statement": "false claim"}]}]
        scenarios = [{"scenario_text": "The truth."}]

        result = nli_hallucination_rate(predictions, scenarios, checker=mock_checker)
        assert result["nli_contradiction_rate"] == 1.0
        assert result["nli_hallucination_rate"] == 1.0

    def test_mixed_results(self, mock_checker):
        call_count = 0

        def side_effect(premise, claim):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"label": "entailment", "scores": {}, "hallucinated": False}
            elif call_count == 2:
                return {"label": "contradiction", "scores": {}, "hallucinated": True}
            else:
                return {"label": "neutral", "scores": {}, "hallucinated": False}

        mock_checker.check_claim.side_effect = side_effect

        predictions = [
            {"claims": [{"statement": "c1"}, {"statement": "c2"}, {"statement": "c3"}]}
        ]
        scenarios = [{"scenario_text": "premise"}]

        result = nli_hallucination_rate(predictions, scenarios, checker=mock_checker)
        assert result["total_claims_checked"] == 3
        assert abs(result["nli_entailment_rate"] - 1 / 3) < 1e-6
        assert abs(result["nli_contradiction_rate"] - 1 / 3) < 1e-6
        # hallucination = contradiction + neutral = 2/3
        assert abs(result["nli_hallucination_rate"] - 2 / 3) < 1e-6

    def test_no_claims_fallback_to_primary_cause(self, mock_checker):
        mock_checker.check_claim.return_value = {
            "label": "entailment",
            "scores": {},
            "hallucinated": False,
        }
        predictions = [{"primary_cause": "Sicherheitsabstand"}]
        scenarios = [{"scenario_text": "Ein Auffahrunfall."}]

        result = nli_hallucination_rate(predictions, scenarios, checker=mock_checker)
        assert result["total_claims_checked"] == 1

    def test_empty_scenario_text_uses_expected_claims(self, mock_checker):
        mock_checker.check_claim.return_value = {
            "label": "entailment",
            "scores": {},
            "hallucinated": False,
        }
        predictions = [{"claims": [{"statement": "something"}]}]
        scenarios = [
            {
                "scenario_text": "",
                "ground_truth": {
                    "expected_claims": ["Claim A", "Claim B"],
                },
            }
        ]

        result = nli_hallucination_rate(predictions, scenarios, checker=mock_checker)
        assert result["total_claims_checked"] == 1
        # Premise should be built from expected_claims
        call_args = mock_checker.check_claim.call_args
        assert "Claim A" in call_args[0][0]

    def test_empty_inputs(self, mock_checker):
        result = nli_hallucination_rate([], [], checker=mock_checker)
        assert result["total_claims_checked"] == 0
        assert result["nli_hallucination_rate"] == 0.0
