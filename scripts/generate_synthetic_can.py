"""Generate synthetic CAN bus data from Autobahn-style scenarios.

NOTE: All data produced by this script is **fully synthetic**. Speed profiles,
braking events, and steering inputs are generated from parametric models with
Gaussian noise — they do not originate from real vehicle measurements. CAN IDs
and encoding conventions follow the Haftung_AI internal specification
(0x100 = speed at 0.1 km/h resolution) and do not correspond to any production
vehicle DBC file. Results derived from this data should be interpreted
accordingly. See README.md § Limitations for details.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScenarioParams:
    """Parameters for a synthetic accident scenario."""

    name: str
    duration_s: float
    initial_speed_kmh: float
    braking_start_s: float
    braking_decel_ms2: float
    steering_event_s: float | None = None
    steering_angle_deg: float = 0.0
    impact_time_s: float | None = None


# Pre-defined scenarios matching the 6 accident categories
SCENARIOS: dict[str, ScenarioParams] = {
    "rear_end": ScenarioParams(
        name="rear_end",
        duration_s=10.0,
        initial_speed_kmh=60.0,
        braking_start_s=7.0,
        braking_decel_ms2=-8.0,
        impact_time_s=9.2,
    ),
    "side_collision": ScenarioParams(
        name="side_collision",
        duration_s=8.0,
        initial_speed_kmh=40.0,
        braking_start_s=5.5,
        braking_decel_ms2=-6.0,
        steering_event_s=5.0,
        steering_angle_deg=25.0,
        impact_time_s=6.8,
    ),
    "head_on": ScenarioParams(
        name="head_on",
        duration_s=6.0,
        initial_speed_kmh=80.0,
        braking_start_s=3.5,
        braking_decel_ms2=-9.5,
        steering_event_s=3.2,
        steering_angle_deg=-15.0,
        impact_time_s=5.0,
    ),
    "intersection": ScenarioParams(
        name="intersection",
        duration_s=8.0,
        initial_speed_kmh=35.0,
        braking_start_s=6.0,
        braking_decel_ms2=-5.0,
        steering_event_s=5.5,
        steering_angle_deg=30.0,
        impact_time_s=7.0,
    ),
    "pedestrian": ScenarioParams(
        name="pedestrian",
        duration_s=6.0,
        initial_speed_kmh=30.0,
        braking_start_s=3.0,
        braking_decel_ms2=-9.0,
        impact_time_s=4.5,
    ),
    "single_vehicle": ScenarioParams(
        name="single_vehicle",
        duration_s=10.0,
        initial_speed_kmh=100.0,
        braking_start_s=6.0,
        braking_decel_ms2=-4.0,
        steering_event_s=5.5,
        steering_angle_deg=45.0,
        impact_time_s=8.0,
    ),
}

# CAN IDs for different signal types (must match telemetry_agent.py conventions)
CAN_IDS = {
    "speed": "0x100",
    "brake": "0x301",
    "steering": "0x180",
    "acceleration": "0x202",
    "yaw_rate": "0x181",
}

SAMPLE_RATE_HZ = 100  # 100 Hz CAN bus


def generate_speed_profile(scenario: ScenarioParams, dt: float) -> list[float]:
    """Generate speed values (km/h) for the scenario timeline."""
    n_samples = int(scenario.duration_s / dt)
    speeds: list[float] = []
    current_speed = scenario.initial_speed_kmh

    for i in range(n_samples):
        t = i * dt
        if t >= scenario.braking_start_s:
            # Apply deceleration (convert m/s² to km/h per step)
            decel_kmh = scenario.braking_decel_ms2 * 3.6 * dt
            current_speed = max(0.0, current_speed + decel_kmh)

        # Add noise
        noise = random.gauss(0, 0.3)
        speeds.append(max(0.0, current_speed + noise))

    return speeds


def generate_steering_profile(scenario: ScenarioParams, dt: float) -> list[float]:
    """Generate steering angle values (degrees) for the scenario."""
    n_samples = int(scenario.duration_s / dt)
    angles: list[float] = []

    for i in range(n_samples):
        t = i * dt
        angle = 0.0

        if scenario.steering_event_s is not None and t >= scenario.steering_event_s:
            elapsed = t - scenario.steering_event_s
            # Smooth sigmoid-like steering
            transition = 1.0 / (1.0 + math.exp(-5 * (elapsed - 0.5)))
            angle = scenario.steering_angle_deg * transition

        noise = random.gauss(0, 0.5)
        angles.append(angle + noise)

    return angles


def speed_to_can_data(speed_kmh: float) -> str:
    """Encode speed as 2-byte hex CAN data (0.1 km/h resolution)."""
    speed_raw = int(max(0, speed_kmh) * 10)
    speed_raw = min(speed_raw, 0xFFFF)
    return f"{(speed_raw >> 8) & 0xFF:02X} {speed_raw & 0xFF:02X} 00 00 00 00 00 00"


def steering_to_can_data(angle_deg: float) -> str:
    """Encode steering angle as signed 2-byte hex."""
    angle_raw = int(angle_deg * 10) + 0x7FFF
    angle_raw = max(0, min(0xFFFF, angle_raw))
    return f"{(angle_raw >> 8) & 0xFF:02X} {angle_raw & 0xFF:02X} 00 00 00 00 00 00"


def brake_to_can_data(braking: bool, pressure_pct: float) -> str:
    """Encode brake status and pressure."""
    status = 0x01 if braking else 0x00
    pressure = int(max(0, min(100, pressure_pct)))
    return f"{status:02X} {pressure:02X} 00 00 00 00 00 00"


def generate_can_csv(
    scenario: ScenarioParams,
    output_path: Path,
    variation: int = 0,
) -> None:
    """Generate a complete CAN bus CSV log for a scenario.

    Args:
        scenario: Scenario parameters.
        output_path: Path to write CSV file.
        variation: Random seed offset for per-sample variation.
    """
    random.seed(42 + variation)
    dt = 1.0 / SAMPLE_RATE_HZ

    speeds = generate_speed_profile(scenario, dt)
    steerings = generate_steering_profile(scenario, dt)

    rows: list[dict[str, str]] = []

    for i, (speed, steering) in enumerate(zip(speeds, steerings)):
        t = i * dt

        # Speed message every 10ms
        if i % 1 == 0:
            rows.append({
                "timestamp": f"{t:.4f}",
                "arbitration_id": CAN_IDS["speed"],
                "data": speed_to_can_data(speed),
                "channel": "0",
            })

        # Steering message every 20ms
        if i % 2 == 0:
            rows.append({
                "timestamp": f"{t:.4f}",
                "arbitration_id": CAN_IDS["steering"],
                "data": steering_to_can_data(steering),
                "channel": "0",
            })

        # Brake message every 10ms
        if i % 1 == 0:
            braking = t >= scenario.braking_start_s
            pressure = 0.0
            if braking:
                elapsed = t - scenario.braking_start_s
                pressure = min(100.0, elapsed * 50.0)
            rows.append({
                "timestamp": f"{t:.4f}",
                "arbitration_id": CAN_IDS["brake"],
                "data": brake_to_can_data(braking, pressure),
                "channel": "0",
            })

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "arbitration_id", "data", "channel"])
        writer.writeheader()
        writer.writerows(rows)


def generate_all(output_base: Path) -> None:
    """Generate CAN logs for all 30 accidents."""
    for category, scenario in SCENARIOS.items():
        for i in range(1, 6):
            accident_id = f"{category}_{i:03d}"
            output_path = output_base / accident_id / "can_log.csv"
            generate_can_csv(scenario, output_path, variation=i)
            print(f"Generated: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic CAN data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation" / "dataset" / "accidents",
    )
    parser.add_argument("--category", choices=list(SCENARIOS.keys()), default=None)
    args = parser.parse_args()

    if args.category:
        scenario = SCENARIOS[args.category]
        for i in range(1, 6):
            path = args.output / f"{args.category}_{i:03d}" / "can_log.csv"
            generate_can_csv(scenario, path, variation=i)
            print(f"Generated: {path}")
    else:
        generate_all(args.output)
