"""Telemetry Agent — wraps CAN parser pipeline."""
from __future__ import annotations

import logging

from haftung_ai.types.state import HaftungState, validate_state

logger = logging.getLogger(__name__)

SYNTHETIC_DATA_NOTICE = (
    "Telemetry data is fully synthetic (parametric speed/braking/steering "
    "profiles with Gaussian noise). CAN IDs follow Haftung_AI internal "
    "conventions and do not correspond to production vehicle DBC files. "
    "See README.md § Limitations."
)


class TelemetryAgent:
    """Processes CAN bus logs into speed profiles, braking events, and ego states."""

    def __call__(self, state: HaftungState) -> HaftungState:
        missing = validate_state(state, ["can_log_path"])
        if missing:
            state.setdefault("errors", []).append(f"TelemetryAgent: missing {missing}")
            return state

        from haftung_ai.telemetry.anomaly_detector import AnomalyDetector
        from haftung_ai.telemetry.can_parser import CANParser
        from haftung_ai.telemetry.ego_reconstructor import EgoReconstructor
        from haftung_ai.telemetry.speed_profile import SpeedProfiler
        from haftung_ai.types.telemetry import SpeedRecord

        parser = CANParser()
        messages = parser.parse(state["can_log_path"])

        # Extract speed records from CAN messages
        # Convention: arbitration_id 0x100 = speed, data[0:2] = speed in 0.1 km/h
        speed_records: list[SpeedRecord] = []
        for msg in messages:
            if msg.arbitration_id == 0x100 and len(msg.data) >= 2:
                speed_raw = int.from_bytes(msg.data[:2], byteorder="big")
                speed_kmh = speed_raw * 0.1
                speed_mps = speed_kmh / 3.6
                speed_records.append(SpeedRecord(timestamp=msg.timestamp, speed_mps=speed_mps, speed_kmh=speed_kmh))

        profiler = SpeedProfiler()
        profile = profiler.build_profile(speed_records)

        anomaly_det = AnomalyDetector()
        braking_anomalies = anomaly_det.detect_braking_anomalies(speed_records)

        reconstructor = EgoReconstructor()
        ego_states = reconstructor.reconstruct(speed_records)

        state["speed_profile"] = {
            "max_speed_kmh": profile.max_speed_kmh,
            "avg_speed_kmh": profile.avg_speed_kmh,
            "num_records": len(profile.records),
            "braking_events": len(profile.braking_events),
        }
        state["braking_events"] = [
            {
                "start_time": e.start_time,
                "end_time": e.end_time,
                "peak_deceleration": e.peak_deceleration,
                "speed_before": e.speed_before,
                "speed_after": e.speed_after,
            }
            for e in profile.braking_events
        ]
        state["ego_states"] = [
            {"x": e.x, "y": e.y, "yaw": e.yaw, "speed_mps": e.speed_mps, "acceleration": e.acceleration}
            for e in ego_states
        ]
        state["telemetry_summary"] = {
            "max_speed_kmh": profile.max_speed_kmh,
            "emergency_braking": any(a["severity"] == "severe" for a in braking_anomalies),
            "num_braking_events": len(profile.braking_events),
        }

        state["synthetic_data_notice"] = SYNTHETIC_DATA_NOTICE

        logger.info("TelemetryAgent: %d speed records, %d braking events", len(speed_records), len(profile.braking_events))
        return state
