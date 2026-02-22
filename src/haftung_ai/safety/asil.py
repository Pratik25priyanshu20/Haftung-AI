"""ISO 26262 ASIL classification (from Autobahn asil_classifier.py)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ASILLevel(str, Enum):
    QM = "QM"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class ComponentASIL:
    component: str
    level: ASILLevel
    rationale: str


COMPONENT_ASIL_MAP: dict[str, ComponentASIL] = {
    "detection": ComponentASIL("detection", ASILLevel.B, "Object detection failure may cause late braking"),
    "tracking": ComponentASIL("tracking", ASILLevel.B, "Track loss can cause phantom braking or missed obstacles"),
    "lane_detection": ComponentASIL("lane_detection", ASILLevel.A, "Lane departure is comfort, not safety-critical"),
    "fcw": ComponentASIL("fcw", ASILLevel.C, "FCW false negative may lead to rear-end collision"),
    "ttc": ComponentASIL("ttc", ASILLevel.C, "TTC underestimate causes late warning"),
    "bsd": ComponentASIL("bsd", ASILLevel.B, "Blind spot miss causes unsafe lane change"),
    "depth_estimation": ComponentASIL("depth_estimation", ASILLevel.A, "Depth aids distance but is not sole source"),
    "lidar_processing": ComponentASIL("lidar_processing", ASILLevel.B, "LIDAR data loss reduces 3D perception"),
    "fusion": ComponentASIL("fusion", ASILLevel.C, "Fusion errors propagate to all downstream safety"),
    "controller": ComponentASIL("controller", ASILLevel.D, "Control output directly affects vehicle behavior"),
}

_ESCALATION_MAP: dict[ASILLevel, str] = {
    ASILLevel.QM: "none",
    ASILLevel.A: "monitoring",
    ASILLevel.B: "monitoring",
    ASILLevel.C: "redundant",
    ASILLevel.D: "fail_safe",
}


class ASILClassifier:
    """Lookup and query ASIL assignments."""

    def __init__(self, overrides: dict[str, str] | None = None):
        self._map: dict[str, ComponentASIL] = dict(COMPONENT_ASIL_MAP)
        if overrides:
            for comp, level_str in overrides.items():
                try:
                    level = ASILLevel(level_str)
                except ValueError:
                    logger.warning("Invalid ASIL override for %s: %s", comp, level_str)
                    continue
                if comp in self._map:
                    self._map[comp] = ComponentASIL(comp, level, self._map[comp].rationale)
                else:
                    self._map[comp] = ComponentASIL(comp, level, "User override")

    def get_level(self, component: str) -> ASILLevel:
        entry = self._map.get(component)
        if entry is None:
            return ASILLevel.QM
        return entry.level

    def requires_redundancy(self, component: str) -> bool:
        return self.get_level(component) in (ASILLevel.C, ASILLevel.D)

    def escalation_level(self, component: str) -> str:
        level = self.get_level(component)
        return _ESCALATION_MAP.get(level, "none")

    def get_all_assignments(self) -> dict[str, ComponentASIL]:
        return dict(self._map)
