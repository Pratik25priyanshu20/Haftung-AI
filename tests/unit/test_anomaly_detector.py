"""Tests for braking/steering anomaly detection."""
from haftung_ai.telemetry.anomaly_detector import AnomalyDetector
from haftung_ai.types.telemetry import SpeedRecord


def test_no_braking_anomalies_constant_speed():
    records = [
        SpeedRecord(timestamp=i * 0.1, speed_mps=10.0, speed_kmh=36.0)
        for i in range(10)
    ]
    detector = AnomalyDetector(braking_threshold_ms2=-5.0)
    anomalies = detector.detect_braking_anomalies(records)
    assert len(anomalies) == 0


def test_braking_anomaly_detected():
    records = [
        SpeedRecord(timestamp=0.0, speed_mps=20.0, speed_kmh=72.0),
        SpeedRecord(timestamp=0.1, speed_mps=19.0, speed_kmh=68.4),
        SpeedRecord(timestamp=0.2, speed_mps=12.0, speed_kmh=43.2),  # -70 m/s²
    ]
    detector = AnomalyDetector(braking_threshold_ms2=-5.0)
    anomalies = detector.detect_braking_anomalies(records)
    assert len(anomalies) >= 1
    assert anomalies[0]["type"] == "emergency_braking"


def test_braking_anomaly_severity():
    records = [
        SpeedRecord(timestamp=0.0, speed_mps=20.0, speed_kmh=72.0),
        SpeedRecord(timestamp=0.1, speed_mps=10.0, speed_kmh=36.0),  # -100 m/s²
    ]
    detector = AnomalyDetector(braking_threshold_ms2=-5.0)
    anomalies = detector.detect_braking_anomalies(records)
    assert anomalies[0]["severity"] == "severe"


def test_steering_anomaly_detected():
    steering_data = [
        {"timestamp": 0.0, "angle": 0.0},
        {"timestamp": 0.1, "angle": 5.0},  # 50 deg/s
    ]
    detector = AnomalyDetector(steering_rate_threshold_degs=45.0)
    events = detector.detect_steering_anomalies(steering_data)
    assert len(events) >= 1
    assert events[0].steering_rate > 45.0


def test_no_steering_anomaly_slow_turn():
    steering_data = [
        {"timestamp": 0.0, "angle": 0.0},
        {"timestamp": 1.0, "angle": 10.0},  # 10 deg/s
    ]
    detector = AnomalyDetector(steering_rate_threshold_degs=45.0)
    events = detector.detect_steering_anomalies(steering_data)
    assert len(events) == 0


def test_empty_inputs():
    detector = AnomalyDetector()
    assert detector.detect_braking_anomalies([]) == []
    assert detector.detect_steering_anomalies([]) == []
