"""Tests for impact detector."""
from dataclasses import dataclass

from haftung_ai.perception.impact_detector import ImpactDetector


@dataclass
class MockTrack:
    track_id: int
    x: float | None = None
    y: float | None = None
    vx: float | None = None
    vy: float | None = None
    ttc: float | None = None


def test_no_impact_far_tracks():
    detector = ImpactDetector(min_distance_threshold_m=2.0, confirmation_frames=1)
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0),
        MockTrack(track_id=2, x=50.0, y=0.0),
    ]
    event = detector.check_frame(tracks, frame_id=0)
    assert event is None


def test_impact_close_tracks():
    detector = ImpactDetector(min_distance_threshold_m=2.0, confirmation_frames=1)
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0),
        MockTrack(track_id=2, x=1.0, y=0.0),
    ]
    event = detector.check_frame(tracks, frame_id=0)
    assert event is not None
    assert event.track_a == 1
    assert event.track_b == 2
    assert event.distance_m == 1.0


def test_impact_requires_confirmation():
    detector = ImpactDetector(min_distance_threshold_m=2.0, confirmation_frames=3)
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0),
        MockTrack(track_id=2, x=1.5, y=0.0),
    ]
    assert detector.check_frame(tracks, frame_id=0) is None
    assert detector.check_frame(tracks, frame_id=1) is None
    event = detector.check_frame(tracks, frame_id=2)
    assert event is not None
    assert event.frame_id == 2


def test_impact_ttc_critical():
    detector = ImpactDetector(ttc_threshold=0.1, min_distance_threshold_m=1.0, confirmation_frames=1)
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0, ttc=0.05),
        MockTrack(track_id=2, x=5.0, y=0.0, ttc=0.05),
    ]
    event = detector.check_frame(tracks, frame_id=0)
    assert event is not None


def test_reset():
    detector = ImpactDetector(confirmation_frames=3)
    tracks = [
        MockTrack(track_id=1, x=0.0, y=0.0),
        MockTrack(track_id=2, x=1.0, y=0.0),
    ]
    detector.check_frame(tracks, frame_id=0)
    detector.reset()
    assert len(detector._candidate_impacts) == 0


def test_empty_tracks():
    detector = ImpactDetector()
    event = detector.check_frame([], frame_id=0)
    assert event is None
