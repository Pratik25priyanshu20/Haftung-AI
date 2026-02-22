"""Tests for Track dataclass."""
from haftung_ai.types.track import Track


def test_track_defaults():
    t = Track(track_id=1, bbox_xyxy=(0, 0, 100, 100), class_name="car", conf=0.9)
    assert t.age == 0
    assert t.is_confirmed is True
    assert t.x is None
    assert t.vx is None
    assert t.ttc is None


def test_track_with_world_coords():
    t = Track(
        track_id=5,
        bbox_xyxy=(50, 50, 150, 150),
        class_name="truck",
        conf=0.8,
        x=15.0,
        y=25.0,
        vx=3.0,
        vy=-1.0,
        ttc=2.5,
    )
    assert t.x == 15.0
    assert t.vx == 3.0
    assert t.ttc == 2.5


def test_track_velocity():
    t = Track(
        track_id=1,
        bbox_xyxy=(0, 0, 50, 50),
        class_name="car",
        conf=0.9,
        velocity_px_per_frame=(2.5, -1.0),
    )
    assert t.velocity_px_per_frame == (2.5, -1.0)
