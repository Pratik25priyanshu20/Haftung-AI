"""Reconstruct EgoState from CAN bus telemetry."""
from __future__ import annotations

import math

from haftung_ai.types.ego import EgoState
from haftung_ai.types.telemetry import SpeedRecord


class EgoReconstructor:
    """Reconstruct ego vehicle trajectory from speed records."""

    def __init__(self, initial_yaw: float = 0.0):
        self.initial_yaw = initial_yaw

    def reconstruct(self, records: list[SpeedRecord], yaw_rates: list[float] | None = None) -> list[EgoState]:
        """Reconstruct EgoState sequence from speed records.

        Args:
            records: Speed measurements over time.
            yaw_rates: Optional yaw rate (rad/s) per time step.

        Returns:
            List of EgoState for each time step.
        """
        if not records:
            return []

        records = sorted(records, key=lambda r: r.timestamp)
        states: list[EgoState] = []

        x, y = 0.0, 0.0
        yaw = self.initial_yaw

        for i, rec in enumerate(records):
            yaw_rate = yaw_rates[i] if yaw_rates and i < len(yaw_rates) else 0.0

            if i > 0:
                dt = rec.timestamp - records[i - 1].timestamp
                if dt > 0:
                    accel = (rec.speed_mps - records[i - 1].speed_mps) / dt
                    yaw += yaw_rate * dt
                    x += rec.speed_mps * math.cos(yaw) * dt
                    y += rec.speed_mps * math.sin(yaw) * dt
                else:
                    accel = 0.0
            else:
                accel = 0.0

            states.append(
                EgoState(
                    x=x,
                    y=y,
                    yaw=yaw,
                    speed=rec.speed_kmh,
                    speed_mps=rec.speed_mps,
                    yaw_rate=yaw_rate,
                    acceleration=accel,
                )
            )

        return states
