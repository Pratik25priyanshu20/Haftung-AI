"""Scene graph builder from tracked objects."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneRelation:
    """Spatial/temporal relation between two tracked objects."""

    track_a: int
    track_b: int
    relation_type: str  # "approaching", "following", "adjacent", "diverging"
    distance_m: float
    closing_rate: float  # m/s (positive = approaching)


@dataclass
class SceneGraph:
    """Graph of tracked objects and their spatial relations."""

    frame_id: int = 0
    timestamp: float = 0.0
    nodes: dict[int, dict[str, Any]] = field(default_factory=dict)  # track_id -> attributes
    edges: list[SceneRelation] = field(default_factory=list)


class SceneGraphBuilder:
    """Builds scene graphs from tracked objects per frame."""

    def __init__(self, proximity_threshold_m: float = 50.0):
        self.proximity_threshold_m = proximity_threshold_m

    def build(self, tracks: list[Any], frame_id: int = 0, timestamp: float = 0.0) -> SceneGraph:
        graph = SceneGraph(frame_id=frame_id, timestamp=timestamp)

        for trk in tracks:
            tid = getattr(trk, "track_id", None)
            if tid is None:
                continue
            graph.nodes[tid] = {
                "class_name": getattr(trk, "class_name", "unknown"),
                "x": getattr(trk, "x", None),
                "y": getattr(trk, "y", None),
                "vx": getattr(trk, "vx", None),
                "vy": getattr(trk, "vy", None),
                "ttc": getattr(trk, "ttc", None),
            }

        track_list = [t for t in tracks if getattr(t, "x", None) is not None]
        n = len(track_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = track_list[i], track_list[j]
                dx = (b.x or 0) - (a.x or 0)
                dy = (b.y or 0) - (a.y or 0)
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > self.proximity_threshold_m:
                    continue

                dvx = (b.vx or 0) - (a.vx or 0)
                dvy = (b.vy or 0) - (a.vy or 0)
                closing_rate = 0.0
                if dist > 1e-6:
                    closing_rate = -(dx * dvx + dy * dvy) / dist  # positive = approaching

                if closing_rate > 0.5:
                    relation_type = "approaching"
                elif closing_rate < -0.5:
                    relation_type = "diverging"
                elif abs(dy) < 2.0 and dx > 0:
                    relation_type = "following"
                else:
                    relation_type = "adjacent"

                graph.edges.append(
                    SceneRelation(
                        track_a=a.track_id,
                        track_b=b.track_id,
                        relation_type=relation_type,
                        distance_m=dist,
                        closing_rate=closing_rate,
                    )
                )

        return graph
