"""Shared test fixtures for Haftung_AI."""
from __future__ import annotations

import numpy as np
import pytest

from haftung_ai.types.detection import Detection
from haftung_ai.types.ego import EgoState
from haftung_ai.types.telemetry import CANMessage, SpeedRecord
from haftung_ai.types.track import Track


@pytest.fixture
def sample_detection() -> Detection:
    return Detection(x1=100, y1=200, x2=300, y2=400, conf=0.85, class_id=2, class_name="car")


@pytest.fixture
def sample_track() -> Track:
    return Track(
        track_id=1,
        bbox_xyxy=(100, 200, 300, 400),
        class_name="car",
        conf=0.85,
        age=5,
        x=10.0,
        y=20.0,
        vx=5.0,
        vy=-2.0,
    )


@pytest.fixture
def sample_ego_state() -> EgoState:
    return EgoState(x=0.0, y=0.0, yaw=0.0, speed=50.0, speed_mps=13.89, acceleration=-2.0)


@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_can_messages() -> list[CANMessage]:
    return [
        CANMessage(timestamp=0.0, arbitration_id=0x100, data=b"\x00\x32"),
        CANMessage(timestamp=0.1, arbitration_id=0x100, data=b"\x00\x30"),
        CANMessage(timestamp=0.2, arbitration_id=0x100, data=b"\x00\x28"),
    ]


@pytest.fixture
def sample_speed_records() -> list[SpeedRecord]:
    return [
        SpeedRecord(timestamp=0.0, speed_mps=13.89, speed_kmh=50.0),
        SpeedRecord(timestamp=0.1, speed_mps=12.50, speed_kmh=45.0),
        SpeedRecord(timestamp=0.2, speed_mps=10.00, speed_kmh=36.0),
        SpeedRecord(timestamp=0.3, speed_mps=5.00, speed_kmh=18.0),
        SpeedRecord(timestamp=0.4, speed_mps=0.00, speed_kmh=0.0),
    ]
