"""CAN bus telemetry types for Haftung_AI."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CANMessage:
    """Single CAN bus message."""

    timestamp: float
    arbitration_id: int
    data: bytes
    channel: str = ""


@dataclass
class SpeedRecord:
    """Speed measurement at a point in time."""

    timestamp: float
    speed_mps: float
    speed_kmh: float


@dataclass
class BrakingEvent:
    """Detected braking event from CAN telemetry."""

    start_time: float
    end_time: float
    peak_deceleration: float  # m/s^2 (negative)
    speed_before: float  # m/s
    speed_after: float  # m/s


@dataclass
class SteeringEvent:
    """Detected anomalous steering event."""

    timestamp: float
    steering_angle: float  # degrees
    steering_rate: float  # degrees/s
    severity: str  # "mild", "moderate", "severe"


@dataclass
class SpeedProfile:
    """Complete speed profile from CAN data."""

    records: list[SpeedRecord] = field(default_factory=list)
    braking_events: list[BrakingEvent] = field(default_factory=list)
    steering_events: list[SteeringEvent] = field(default_factory=list)
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0


@dataclass
class TelemetrySummary:
    """Summary of CAN bus telemetry for an incident."""

    speed_profile: SpeedProfile = field(default_factory=SpeedProfile)
    impact_speed_kmh: float | None = None
    emergency_braking: bool = False
    abs_activated: bool = False
    esc_activated: bool = False
    airbag_deployed: bool = False
