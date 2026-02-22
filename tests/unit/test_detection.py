"""Tests for Detection dataclass."""
from haftung_ai.types.detection import Detection


def test_detection_bbox():
    d = Detection(x1=10, y1=20, x2=110, y2=120, conf=0.9, class_id=2, class_name="car")
    assert d.bbox == (10, 20, 110, 120)


def test_detection_width_height():
    d = Detection(x1=0, y1=0, x2=100, y2=50, conf=0.5, class_id=0, class_name="person")
    assert d.width == 100
    assert d.height == 50


def test_detection_center():
    d = Detection(x1=0, y1=0, x2=100, y2=100, conf=0.5, class_id=0, class_name="car")
    assert d.center == (50.0, 50.0)


def test_detection_score():
    d = Detection(x1=0, y1=0, x2=10, y2=10, conf=0.75, class_id=1, class_name="truck")
    assert d.score == 0.75


def test_detection_label():
    d = Detection(x1=0, y1=0, x2=10, y2=10, conf=0.5, class_id=3, class_name="bicycle")
    assert d.label == "bicycle"
