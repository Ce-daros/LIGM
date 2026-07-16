import pytest

from ligm.online import compare_local_recovery


def report(local_accuracy: float) -> dict:
    return {"buckets": {"local": {"accuracy": local_accuracy}}}


def test_local_guard_accepts_exact_negative_half_point_boundary() -> None:
    comparison = compare_local_recovery(report(0.795), report(0.8), 0.005)

    assert comparison["local_delta"] == pytest.approx(-0.005)
    assert comparison["passed"]


def test_local_guard_stops_below_negative_half_point_boundary() -> None:
    comparison = compare_local_recovery(report(0.7949), report(0.8), 0.005)

    assert not comparison["passed"]
