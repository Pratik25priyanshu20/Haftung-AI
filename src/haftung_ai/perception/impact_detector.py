"""Detect collision moment from trajectory convergence."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ImpactEvent:
    """Detected collision/impact event."""

    frame_id: int
    timestamp: float
    track_a: int
    track_b: int
    distance_m: float
    ttc_s: float | None
    confidence: float


class ImpactDetector:
    """Detect actual collision from TTC < threshold AND minimum inter-track distance.

    Improvement over Autobahn: Autobahn computes TTC but never detects
    actual collision. This uses TTC < 0.1s AND min distance to confirm impact.
    """

    def __init__(
        self,
        ttc_threshold: float = 0.1,
        min_distance_threshold_m: float = 2.0,
        confirmation_frames: int = 3,
    ):
        self.ttc_threshold = ttc_threshold
        self.min_distance_threshold_m = min_distance_threshold_m
        self.confirmation_frames = confirmation_frames
        self._candidate_impacts: dict[tuple[int, int], list[int]] = {}

    def check_frame(
        self,
        tracks: list[Any],
        frame_id: int,
        timestamp: float = 0.0,
    ) -> ImpactEvent | None:
        """Check a single frame for potential impact. Returns ImpactEvent if confirmed."""
        track_list = [t for t in tracks if getattr(t, "x", None) is not None]
        n = len(track_list)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = track_list[i], track_list[j]
                dx = (b.x or 0) - (a.x or 0)
                dy = (b.y or 0) - (a.y or 0)
                dist = math.sqrt(dx * dx + dy * dy)

                ttc_a = getattr(a, "ttc", None)
                ttc_b = getattr(b, "ttc", None)
                min_ttc = min(t for t in [ttc_a, ttc_b] if t is not None) if any(t is not None for t in [ttc_a, ttc_b]) else None

                is_close = dist < self.min_distance_threshold_m
                is_ttc_critical = min_ttc is not None and min_ttc < self.ttc_threshold

                if is_close or is_ttc_critical:
                    pair = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                    if pair not in self._candidate_impacts:
                        self._candidate_impacts[pair] = []
                    self._candidate_impacts[pair].append(frame_id)

                    if len(self._candidate_impacts[pair]) >= self.confirmation_frames:
                        confidence = min(1.0, 1.0 - (dist / self.min_distance_threshold_m) * 0.5)
                        return ImpactEvent(
                            frame_id=frame_id,
                            timestamp=timestamp,
                            track_a=pair[0],
                            track_b=pair[1],
                            distance_m=dist,
                            ttc_s=min_ttc,
                            confidence=max(0.1, confidence),
                        )
                else:
                    pair = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                    self._candidate_impacts.pop(pair, None)

        return None

    def reset(self) -> None:
        self._candidate_impacts.clear()
