"""Tests for ASIL classifier."""
from haftung_ai.safety.asil import ASILClassifier, ASILLevel


def test_default_detection_level():
    classifier = ASILClassifier()
    assert classifier.get_level("detection") == ASILLevel.B


def test_controller_is_d():
    classifier = ASILClassifier()
    assert classifier.get_level("controller") == ASILLevel.D


def test_unknown_component_is_qm():
    classifier = ASILClassifier()
    assert classifier.get_level("nonexistent") == ASILLevel.QM


def test_requires_redundancy():
    classifier = ASILClassifier()
    assert classifier.requires_redundancy("controller") is True
    assert classifier.requires_redundancy("fcw") is True
    assert classifier.requires_redundancy("detection") is False
    assert classifier.requires_redundancy("lane_detection") is False


def test_escalation_level():
    classifier = ASILClassifier()
    assert classifier.escalation_level("controller") == "fail_safe"
    assert classifier.escalation_level("fcw") == "redundant"
    assert classifier.escalation_level("detection") == "monitoring"


def test_override():
    classifier = ASILClassifier(overrides={"detection": "D"})
    assert classifier.get_level("detection") == ASILLevel.D
    assert classifier.requires_redundancy("detection") is True


def test_invalid_override_ignored():
    classifier = ASILClassifier(overrides={"detection": "INVALID"})
    assert classifier.get_level("detection") == ASILLevel.B


def test_get_all_assignments():
    classifier = ASILClassifier()
    assignments = classifier.get_all_assignments()
    assert "detection" in assignments
    assert "controller" in assignments
    assert len(assignments) >= 10
