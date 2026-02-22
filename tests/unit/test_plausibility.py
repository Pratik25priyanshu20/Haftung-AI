"""Tests for plausibility checker."""
from dataclasses import dataclass

from haftung_ai.safety.plausibility import PlausibilityChecker


@dataclass
class MockTrack:
    track_id: int = 1
    vx: float | None = None
    vy: float | None = None
    x: float | None = None
    y: float | None = None


@dataclass
class MockDetection:
    bbox_xyxy: tuple[int, int, int, int] = (0, 0, 10, 10)


def test_no_violations():
    checker = PlausibilityChecker()
    tracks = [MockTrack(track_id=1, vx=5.0, vy=0.0)]
    violations = checker.check(tracks, [MockDetection()])
    assert len(violations) == 0


def test_velocity_violation():
    checker = PlausibilityChecker(max_velocity_kmh=100.0)
    tracks = [MockTrack(track_id=1, vx=30.0, vy=30.0)]  # ~152 km/h
    violations = checker.check(tracks, [])
    velocity_violations = [v for v in violations if v.check_name == "velocity"]
    assert len(velocity_violations) == 1


def test_position_jump_violation():
    checker = PlausibilityChecker(max_position_jump_m=5.0)
    tracks = [MockTrack(track_id=1, x=20.0, y=0.0)]
    prev_tracks = [MockTrack(track_id=1, x=0.0, y=0.0)]
    violations = checker.check(tracks, [], prev_tracks)
    jump_violations = [v for v in violations if v.check_name == "position_jump"]
    assert len(jump_violations) == 1


def test_no_position_jump_normal():
    checker = PlausibilityChecker(max_position_jump_m=5.0)
    tracks = [MockTrack(track_id=1, x=1.0, y=0.0)]
    prev_tracks = [MockTrack(track_id=1, x=0.0, y=0.0)]
    violations = checker.check(tracks, [], prev_tracks)
    jump_violations = [v for v in violations if v.check_name == "position_jump"]
    assert len(jump_violations) == 0


def test_bbox_overlap_violation():
    checker = PlausibilityChecker(max_bbox_overlap=0.5)
    detections = [
        MockDetection(bbox_xyxy=(0, 0, 100, 100)),
        MockDetection(bbox_xyxy=(10, 10, 110, 110)),
    ]
    violations = checker.check([], detections)
    overlap_violations = [v for v in violations if v.check_name == "bbox_overlap"]
    assert len(overlap_violations) == 1


def test_detection_count_violation():
    checker = PlausibilityChecker(max_detection_count=5)
    detections = [MockDetection() for _ in range(10)]
    violations = checker.check([], detections)
    count_violations = [v for v in violations if v.check_name == "detection_count"]
    assert len(count_violations) == 1
