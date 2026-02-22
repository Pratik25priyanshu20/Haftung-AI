"""Tests for TTC computation."""
from haftung_ai.safety.ttc import compute_ttc


def test_basic_ttc():
    ttc = compute_ttc(distance_proxy=10.0, closing_rate=5.0)
    assert ttc == 2.0


def test_ttc_zero_closing_rate():
    ttc = compute_ttc(distance_proxy=10.0, closing_rate=0.0)
    assert ttc is None


def test_ttc_negative_closing_rate():
    ttc = compute_ttc(distance_proxy=10.0, closing_rate=-1.0)
    assert ttc is None


def test_ttc_zero_distance():
    ttc = compute_ttc(distance_proxy=0.0, closing_rate=5.0)
    assert ttc is None


def test_ttc_very_small_values():
    ttc = compute_ttc(distance_proxy=1e-7, closing_rate=1e-7)
    assert ttc is None


def test_ttc_large_distance():
    ttc = compute_ttc(distance_proxy=1000.0, closing_rate=10.0)
    assert ttc == 100.0
