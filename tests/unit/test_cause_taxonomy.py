"""Tests for cause taxonomy classifier."""
from evaluation.metrics.cause_taxonomy import (
    CAUSE_TAXONOMY,
    causation_accuracy_taxonomy,
    classify_cause,
)


class TestClassifyCause:
    def test_following_distance_german(self):
        assert classify_cause("Unzureichender Sicherheitsabstand") == "following_distance"

    def test_following_distance_english(self):
        assert classify_cause("following distance too short") == "following_distance"

    def test_speeding_german(self):
        assert classify_cause("Überhöhte Geschwindigkeit") == "speeding"

    def test_right_of_way(self):
        assert classify_cause("Vorfahrtsverletzung an der Kreuzung") == "right_of_way"

    def test_red_light(self):
        assert classify_cause("Rotlichtverstoß an der Ampel") == "red_light"

    def test_pedestrian_crossing(self):
        assert classify_cause("Fußgänger am Zebrastreifen übersehen") == "pedestrian_crossing"

    def test_distraction(self):
        assert classify_cause("Ablenkung durch Mobiltelefon") == "distraction"

    def test_overtaking(self):
        assert classify_cause("Gefährliches Überholmanöver") == "overtaking"

    def test_unknown_cause(self):
        assert classify_cause("completely unrelated text xyz") == "unknown"

    def test_empty_string(self):
        assert classify_cause("") == "unknown"

    def test_case_insensitive(self):
        assert classify_cause("GESCHWINDIGKEIT ZU HOCH") == "speeding"

    def test_multiple_keywords_picks_best(self):
        # "Auffahrunfall wegen zu geringem Sicherheitsabstand" has multiple following_distance keywords
        result = classify_cause("Auffahrunfall wegen zu geringem Sicherheitsabstand")
        assert result == "following_distance"

    def test_taxonomy_has_20_categories(self):
        assert len(CAUSE_TAXONOMY) == 20

    def test_all_categories_have_keywords(self):
        for cat_id, keywords in CAUSE_TAXONOMY.items():
            assert len(keywords) >= 3, f"{cat_id} has fewer than 3 keywords"


class TestCausationAccuracyTaxonomy:
    def test_perfect_accuracy(self):
        preds = [
            {"primary_cause": "Sicherheitsabstand zu gering"},
            {"primary_cause": "Überhöhte Geschwindigkeit"},
        ]
        gts = [
            {"primary_cause_taxonomy_id": "following_distance", "category": "rear_end"},
            {"primary_cause_taxonomy_id": "speeding", "category": "single_vehicle"},
        ]
        result = causation_accuracy_taxonomy(preds, gts)
        assert result["exact_match"] == 1.0
        assert result["n"] == 2
        assert result["per_category"]["rear_end"] == 1.0
        assert result["per_category"]["single_vehicle"] == 1.0

    def test_zero_accuracy(self):
        preds = [
            {"primary_cause": "Sicherheitsabstand zu gering"},
        ]
        gts = [
            {"primary_cause_taxonomy_id": "speeding", "category": "rear_end"},
        ]
        result = causation_accuracy_taxonomy(preds, gts)
        assert result["exact_match"] == 0.0

    def test_empty_inputs(self):
        result = causation_accuracy_taxonomy([], [])
        assert result["exact_match"] == 0.0
        assert result["n"] == 0

    def test_fallback_to_text_classification(self):
        """When GT has no taxonomy_id, fall back to classifying GT text."""
        preds = [{"primary_cause": "Auffahrunfall wegen Abstand"}]
        gts = [{"primary_cause": "Sicherheitsabstand", "category": "rear_end"}]
        result = causation_accuracy_taxonomy(preds, gts)
        assert result["exact_match"] == 1.0

    def test_partial_accuracy(self):
        preds = [
            {"primary_cause": "Sicherheitsabstand"},
            {"primary_cause": "Rotlichtverstoß"},
            {"primary_cause": "random nonsense xyz"},
        ]
        gts = [
            {"primary_cause_taxonomy_id": "following_distance", "category": "rear_end"},
            {"primary_cause_taxonomy_id": "red_light", "category": "intersection"},
            {"primary_cause_taxonomy_id": "speeding", "category": "single_vehicle"},
        ]
        result = causation_accuracy_taxonomy(preds, gts)
        assert abs(result["exact_match"] - 2 / 3) < 1e-6
