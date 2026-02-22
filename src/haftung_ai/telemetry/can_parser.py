"""Parse CSV/ASC/BLF CAN bus logs."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from haftung_ai.types.telemetry import CANMessage

logger = logging.getLogger(__name__)


class CANParser:
    """Parse CAN bus log files into CANMessage objects."""

    def parse(self, path: str | Path) -> list[CANMessage]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CAN log not found: {path}")

        ext = path.suffix.lower()
        if ext == ".csv":
            return self._parse_csv(path)
        elif ext == ".asc":
            return self._parse_asc(path)
        elif ext == ".blf":
            return self._parse_blf(path)
        else:
            raise ValueError(f"Unsupported CAN format: {ext}")

    def _parse_csv(self, path: Path) -> list[CANMessage]:
        messages: list[CANMessage] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = float(row.get("timestamp", row.get("time", 0)))
                    arb_id_str = row.get("arbitration_id", row.get("id", row.get("can_id", "0")))
                    arb_id = int(arb_id_str, 16) if arb_id_str.startswith("0x") else int(arb_id_str)
                    data_str = row.get("data", row.get("payload", ""))
                    data = bytes.fromhex(data_str.replace(" ", "").replace("0x", "")) if data_str else b""
                    channel = row.get("channel", "")
                    messages.append(CANMessage(timestamp=timestamp, arbitration_id=arb_id, data=data, channel=channel))
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping malformed CAN row: %s", e)
        logger.info("Parsed %d CAN messages from %s", len(messages), path)
        return messages

    def _parse_asc(self, path: Path) -> list[CANMessage]:
        messages: list[CANMessage] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("date") or line.startswith("base"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    timestamp = float(parts[0])
                    arb_id = int(parts[2], 16)
                    dlc = int(parts[5]) if len(parts) > 5 else 0
                    data_bytes = bytes(int(b, 16) for b in parts[6 : 6 + dlc]) if dlc > 0 else b""
                    channel = parts[1] if len(parts) > 1 else ""
                    messages.append(CANMessage(timestamp=timestamp, arbitration_id=arb_id, data=data_bytes, channel=channel))
                except (ValueError, IndexError):
                    continue
        logger.info("Parsed %d CAN messages from ASC %s", len(messages), path)
        return messages

    def _parse_blf(self, path: Path) -> list[CANMessage]:
        try:
            import can

            messages: list[CANMessage] = []
            with can.BLFReader(str(path)) as reader:
                for msg in reader:
                    messages.append(
                        CANMessage(
                            timestamp=msg.timestamp,
                            arbitration_id=msg.arbitration_id,
                            data=bytes(msg.data),
                            channel=str(msg.channel or ""),
                        )
                    )
            logger.info("Parsed %d CAN messages from BLF %s", len(messages), path)
            return messages
        except ImportError:
            raise ImportError("python-can is required for BLF parsing: pip install python-can")
