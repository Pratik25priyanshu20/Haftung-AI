"""Tests for CAN bus parser."""
import pytest

from haftung_ai.telemetry.can_parser import CANParser


@pytest.fixture
def csv_can_file(tmp_path):
    content = (
        "timestamp,arbitration_id,data,channel\n"
        "0.0000,0x201,00 32 00 00 00 00 00 00,0\n"
        "0.0100,0x201,00 30 00 00 00 00 00 00,0\n"
        "0.0200,0x301,01 50 00 00 00 00 00 00,0\n"
    )
    f = tmp_path / "test.csv"
    f.write_text(content)
    return f


@pytest.fixture
def asc_can_file(tmp_path):
    content = (
        "date Mon Jan 01 00:00:00 2024\n"
        "base hex timestamps absolute\n"
        "   0.000000 1  201             Rx   d 8 00 32 00 00 00 00 00 00\n"
        "   0.010000 1  201             Rx   d 8 00 30 00 00 00 00 00 00\n"
    )
    f = tmp_path / "test.asc"
    f.write_text(content)
    return f


def test_parse_csv(csv_can_file):
    parser = CANParser()
    messages = parser.parse(csv_can_file)
    assert len(messages) == 3
    assert messages[0].arbitration_id == 0x201
    assert messages[0].timestamp == 0.0
    assert messages[2].arbitration_id == 0x301


def test_parse_csv_data_bytes(csv_can_file):
    parser = CANParser()
    messages = parser.parse(csv_can_file)
    assert messages[0].data == bytes.fromhex("0032000000000000")


def test_parse_asc(asc_can_file):
    parser = CANParser()
    messages = parser.parse(asc_can_file)
    assert len(messages) == 2
    assert messages[0].arbitration_id == 0x201


def test_parse_nonexistent():
    parser = CANParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.csv")


def test_parse_unsupported_format(tmp_path):
    f = tmp_path / "test.xyz"
    f.write_text("data")
    parser = CANParser()
    with pytest.raises(ValueError, match="Unsupported CAN format"):
        parser.parse(f)


def test_parse_empty_csv(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("timestamp,arbitration_id,data,channel\n")
    parser = CANParser()
    messages = parser.parse(f)
    assert len(messages) == 0
