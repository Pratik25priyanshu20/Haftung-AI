"""Canonical Track type for Haftung_AI (from Autobahn)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    """Unified tracked-object representation."""

    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    class_name: str
    conf: float
    age: int = 0
    is_confirmed: bool = True
    velocity_px_per_frame: tuple[float, float] | None = None

    # World-frame attributes (populated by Kalman)
    x: float | None = None
    y: float | None = None
    vx: float | None = None
    vy: float | None = None
    ttc: float | None = None
    risk: str | None = None
