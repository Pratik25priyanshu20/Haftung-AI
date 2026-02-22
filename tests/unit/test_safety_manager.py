"""Tests for safety manager."""
from haftung_ai.safety.safety_manager import SafetyManager
from haftung_ai.types.safety import SafetyStateEnum


def test_normal_state():
    mgr = SafetyManager()
    output = mgr.evaluate()
    assert output.state == SafetyStateEnum.NORMAL
    assert output.message == "System OK"


def test_fcw_critical():
    mgr = SafetyManager()
    output = mgr.evaluate(fcw_state="CRITICAL", fcw_ttc_s=0.5)
    assert output.state == SafetyStateEnum.CRITICAL
    assert "FCW CRITICAL" in output.message
    assert "TTC=0.50s" in output.message


def test_fcw_warning():
    mgr = SafetyManager()
    output = mgr.evaluate(fcw_state="WARNING", fcw_ttc_s=1.5)
    assert output.state == SafetyStateEnum.WARNING
    assert "FCW WARNING" in output.message


def test_fcw_caution():
    mgr = SafetyManager()
    output = mgr.evaluate(fcw_state="CAUTION")
    assert output.state == SafetyStateEnum.CAUTION


def test_fcw_pre_active():
    mgr = SafetyManager()
    output = mgr.evaluate(fcw_pre_active=True)
    assert output.state == SafetyStateEnum.AWARENESS


def test_ldw_departure():
    mgr = SafetyManager()
    output = mgr.evaluate(lane_ok=True, ldw_departure="LEFT")
    assert output.state == SafetyStateEnum.CAUTION
    assert "LDW LEFT" in output.message


def test_bsd_warning():
    mgr = SafetyManager()
    output = mgr.evaluate(bsd_warnings=[{"side": "left"}])
    assert output.state == SafetyStateEnum.CAUTION
    assert "BSD LEFT" in output.message


def test_evaluate_with_asil_no_violations():
    mgr = SafetyManager()
    output = mgr.evaluate_with_asil()
    assert output.state == SafetyStateEnum.NORMAL


def test_evaluate_with_asil_critical_plausibility():
    from dataclasses import dataclass

    @dataclass
    class MockViolation:
        check_name: str = "velocity"
        description: str = "too fast"
        severity: str = "critical"

    mgr = SafetyManager()
    output = mgr.evaluate_with_asil(plausibility_violations=[MockViolation()])
    assert output.state == SafetyStateEnum.WARNING
    assert "PLAUS CRITICAL" in output.message


def test_color_normal():
    mgr = SafetyManager()
    output = mgr.evaluate()
    assert output.color == (0, 255, 0)
