"""Tests for speed profile extraction."""
from haftung_ai.telemetry.speed_profile import SpeedProfiler
from haftung_ai.types.telemetry import SpeedRecord


def test_empty_records():
    profiler = SpeedProfiler()
    profile = profiler.build_profile([])
    assert profile.max_speed_kmh == 0.0
    assert profile.avg_speed_kmh == 0.0
    assert len(profile.records) == 0


def test_basic_profile():
    records = [
        SpeedRecord(timestamp=0.0, speed_mps=10.0, speed_kmh=36.0),
        SpeedRecord(timestamp=1.0, speed_mps=15.0, speed_kmh=54.0),
        SpeedRecord(timestamp=2.0, speed_mps=20.0, speed_kmh=72.0),
    ]
    profiler = SpeedProfiler()
    profile = profiler.build_profile(records)
    assert profile.max_speed_kmh == 72.0
    assert abs(profile.avg_speed_kmh - 54.0) < 0.01


def test_braking_detection(sample_speed_records):
    profiler = SpeedProfiler(braking_threshold_ms2=-3.0, min_braking_duration_s=0.05)
    profile = profiler.build_profile(sample_speed_records)
    assert len(profile.braking_events) >= 1
    event = profile.braking_events[0]
    assert event.peak_deceleration < -3.0


def test_no_braking_at_constant_speed():
    records = [
        SpeedRecord(timestamp=i * 0.1, speed_mps=10.0, speed_kmh=36.0)
        for i in range(20)
    ]
    profiler = SpeedProfiler(braking_threshold_ms2=-3.0)
    profile = profiler.build_profile(records)
    assert len(profile.braking_events) == 0


def test_braking_event_speed_before_after(sample_speed_records):
    profiler = SpeedProfiler(braking_threshold_ms2=-3.0, min_braking_duration_s=0.05)
    profile = profiler.build_profile(sample_speed_records)
    if profile.braking_events:
        event = profile.braking_events[0]
        assert event.speed_before >= event.speed_after
