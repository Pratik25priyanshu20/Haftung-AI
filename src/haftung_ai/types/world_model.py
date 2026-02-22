"""Stripped WorldModel for Haftung_AI accident analysis (adapted from Autobahn)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WorldModel:
    """Per-frame state for accident analysis.

    Stripped to fields relevant for post-hoc analysis (no LIDAR, radar, control).
    """

    frame_id: int = 0
    timestamp: float = 0.0
    frame: np.ndarray | None = None
    detections: list[Any] = field(default_factory=list)
    tracks: list[Any] = field(default_factory=list)
    trajectories: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sensor_health: dict[str, float] = field(default_factory=dict)

    # Kalman-smoothed predictions
    predictions: dict[int, list[Any]] = field(default_factory=dict)

    @staticmethod
    def ema(prev: float | None, curr: float, alpha: float = 0.8) -> float:
        return curr if prev is None else alpha * curr + (1.0 - alpha) * prev

    def summary(self) -> str:
        return (
            f"frame={self.frame_id} "
            f"t={self.timestamp:.3f}s "
            f"objects={len(self.tracks)} "
            f"detections={len(self.detections)}"
        )
