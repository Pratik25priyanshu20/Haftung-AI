"""Report section types for Haftung_AI Unfallbericht."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AccidentMetadata:
    """Basic accident metadata for the report header."""

    report_id: str = ""
    date: datetime | None = None
    location: str = ""
    weather: str = ""
    road_condition: str = ""
    light_condition: str = ""


@dataclass
class VehicleInfo:
    """Vehicle information for a party."""

    party_id: str = ""
    vehicle_type: str = ""
    license_plate: str = ""
    speed_at_impact_kmh: float | None = None
    direction_of_travel: str = ""
    damage_description: str = ""


@dataclass
class ReportSection:
    """A section of the Unfallbericht."""

    title: str = ""
    content: str = ""
    subsections: list[ReportSection] = field(default_factory=list)


@dataclass
class AccidentReport:
    """Complete German accident report (Unfallbericht)."""

    metadata: AccidentMetadata = field(default_factory=AccidentMetadata)
    vehicles: list[VehicleInfo] = field(default_factory=list)
    unfallhergang: str = ""  # Accident sequence
    unfallursache: str = ""  # Accident cause
    haftungsverteilung: str = ""  # Liability distribution
    schadenbeschreibung: str = ""  # Damage description
    beweismittel: list[str] = field(default_factory=list)  # Evidence
    rechtliche_grundlagen: list[str] = field(default_factory=list)  # Legal basis
    scene_diagram_path: str | None = None
    sections: list[ReportSection] = field(default_factory=list)
