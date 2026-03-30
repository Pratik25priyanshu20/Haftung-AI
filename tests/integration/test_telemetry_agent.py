"""Integration tests for TelemetryAgent with synthetic CAN data."""
from __future__ import annotations

import pytest

from haftung_ai.types.state import HaftungState


@pytest.fixture
def can_csv(tmp_path):
    """Create a minimal synthetic CAN CSV.

    Uses CAN ID 0x100 for speed (matching telemetry_agent convention)
    with 0.1 km/h resolution encoding.
    """
    content = "timestamp,arbitration_id,data,channel\n"
    for i in range(100):
        t = i * 0.01
        speed_raw = int(max(0, 50 - i * 0.5) * 10)
        data = f"{(speed_raw >> 8) & 0xFF:02X} {speed_raw & 0xFF:02X} 00 00 00 00 00 00"
        content += f"{t:.4f},0x100,{data},0\n"
    path = tmp_path / "test_can.csv"
    path.write_text(content)
    return str(path)


@pytest.mark.skipif(True, reason="Requires full telemetry pipeline setup")
class TestTelemetryAgentIntegration:

    def test_processes_can_log(self, can_csv):
        from haftung_ai.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent()
        state: HaftungState = {
            "can_log_path": can_csv,
            "errors": [],
            "warnings": [],
        }
        result = agent(state)
        assert result.get("speed_profile") is not None
