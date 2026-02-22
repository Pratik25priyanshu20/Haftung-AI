"""Braking/steering anomaly detection from CAN telemetry."""
from __future__ import annotations

import logging

from haftung_ai.types.telemetry import SpeedRecord, SteeringEvent

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalous braking and steering events."""

    def __init__(
        self,
        braking_threshold_ms2: float = -5.0,
        steering_rate_threshold_degs: float = 45.0,
    ):
        self.braking_threshold_ms2 = braking_threshold_ms2
        self.steering_rate_threshold_degs = steering_rate_threshold_degs

    def detect_braking_anomalies(self, records: list[SpeedRecord]) -> list[dict]:
        """Detect sudden/emergency braking events."""
        anomalies: list[dict] = []
        if len(records) < 2:
            return anomalies

        for i in range(1, len(records)):
            dt = records[i].timestamp - records[i - 1].timestamp
            if dt <= 0:
                continue
            accel = (records[i].speed_mps - records[i - 1].speed_mps) / dt
            if accel < self.braking_threshold_ms2:
                severity = "severe" if accel < self.braking_threshold_ms2 * 1.5 else "moderate"
                anomalies.append({
                    "timestamp": records[i].timestamp,
                    "deceleration_ms2": accel,
                    "speed_before_kmh": records[i - 1].speed_kmh,
                    "speed_after_kmh": records[i].speed_kmh,
                    "severity": severity,
                    "type": "emergency_braking",
                })

        return anomalies

    def detect_steering_anomalies(self, steering_data: list[dict]) -> list[SteeringEvent]:
        """Detect anomalous steering from steering angle data.

        steering_data: list of {"timestamp": float, "angle": float} dicts.
        """
        events: list[SteeringEvent] = []
        if len(steering_data) < 2:
            return events

        for i in range(1, len(steering_data)):
            dt = steering_data[i]["timestamp"] - steering_data[i - 1]["timestamp"]
            if dt <= 0:
                continue
            rate = abs(steering_data[i]["angle"] - steering_data[i - 1]["angle"]) / dt

            if rate > self.steering_rate_threshold_degs:
                if rate > self.steering_rate_threshold_degs * 2:
                    severity = "severe"
                elif rate > self.steering_rate_threshold_degs * 1.5:
                    severity = "moderate"
                else:
                    severity = "mild"

                events.append(
                    SteeringEvent(
                        timestamp=steering_data[i]["timestamp"],
                        steering_angle=steering_data[i]["angle"],
                        steering_rate=rate,
                        severity=severity,
                    )
                )

        return events
