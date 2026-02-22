"""Speed/acceleration extraction from CAN bus data."""
from __future__ import annotations

import logging

from haftung_ai.types.telemetry import BrakingEvent, SpeedProfile, SpeedRecord

logger = logging.getLogger(__name__)


class SpeedProfiler:
    """Extract speed profile, braking events from speed records."""

    def __init__(
        self,
        braking_threshold_ms2: float = -3.0,
        min_braking_duration_s: float = 0.1,
    ):
        self.braking_threshold_ms2 = braking_threshold_ms2
        self.min_braking_duration_s = min_braking_duration_s

    def build_profile(self, records: list[SpeedRecord]) -> SpeedProfile:
        if not records:
            return SpeedProfile()

        records = sorted(records, key=lambda r: r.timestamp)
        speeds_kmh = [r.speed_kmh for r in records]
        max_speed = max(speeds_kmh) if speeds_kmh else 0.0
        avg_speed = sum(speeds_kmh) / len(speeds_kmh) if speeds_kmh else 0.0

        braking_events = self._detect_braking(records)

        return SpeedProfile(
            records=records,
            braking_events=braking_events,
            max_speed_kmh=max_speed,
            avg_speed_kmh=avg_speed,
        )

    def _detect_braking(self, records: list[SpeedRecord]) -> list[BrakingEvent]:
        events: list[BrakingEvent] = []
        if len(records) < 2:
            return events

        in_braking = False
        start_idx = 0
        peak_decel = 0.0

        for i in range(1, len(records)):
            dt = records[i].timestamp - records[i - 1].timestamp
            if dt <= 0:
                continue
            accel = (records[i].speed_mps - records[i - 1].speed_mps) / dt

            if accel < self.braking_threshold_ms2:
                if not in_braking:
                    in_braking = True
                    start_idx = i - 1
                    peak_decel = accel
                else:
                    peak_decel = min(peak_decel, accel)
            else:
                if in_braking:
                    duration = records[i - 1].timestamp - records[start_idx].timestamp
                    if duration >= self.min_braking_duration_s:
                        events.append(
                            BrakingEvent(
                                start_time=records[start_idx].timestamp,
                                end_time=records[i - 1].timestamp,
                                peak_deceleration=peak_decel,
                                speed_before=records[start_idx].speed_mps,
                                speed_after=records[i - 1].speed_mps,
                            )
                        )
                    in_braking = False

        if in_braking:
            duration = records[-1].timestamp - records[start_idx].timestamp
            if duration >= self.min_braking_duration_s:
                events.append(
                    BrakingEvent(
                        start_time=records[start_idx].timestamp,
                        end_time=records[-1].timestamp,
                        peak_deceleration=peak_decel,
                        speed_before=records[start_idx].speed_mps,
                        speed_after=records[-1].speed_mps,
                    )
                )

        return events
