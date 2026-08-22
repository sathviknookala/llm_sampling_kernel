import math

import pytest

from benchmarks.amdahl_probe import amdahl_gain_pct
from benchmarks.harness import percentile


def test_percentile_endpoints():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0.0) == 1.0
    assert percentile(xs, 1.0) == 5.0
    assert percentile(xs, 0.5) == 3.0


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)


def test_percentile_single_and_empty():
    assert percentile([7.0], 0.95) == 7.0
    assert math.isnan(percentile([], 0.5))


def test_percentile_is_order_independent():
    assert percentile([5.0, 1.0, 3.0], 0.5) == percentile([1.0, 3.0, 5.0], 0.5)


@pytest.mark.parametrize(
    "frac,speedup,expected",
    [
        (0.10, 2, 5.0),
        (0.10, 10, 9.0),
        (0.10, float("inf"), 10.0),
        (0.01, 10, 0.9),
        (1.0, 2, 50.0),
    ],
)
def test_amdahl_gain(frac, speedup, expected):
    assert amdahl_gain_pct(frac, speedup) == pytest.approx(expected)


def test_amdahl_gain_is_bounded_by_the_fraction():
    for frac in (0.001, 0.05, 0.5):
        for s in (2, 5, 10, 100):
            assert amdahl_gain_pct(frac, s) < amdahl_gain_pct(frac, float("inf"))
        assert amdahl_gain_pct(frac, float("inf")) == pytest.approx(100 * frac)


def test_amdahl_gain_is_monotone_in_speedup():
    gains = [amdahl_gain_pct(0.2, s) for s in (1, 2, 5, 10, 1000)]
    assert gains == sorted(gains)
    assert gains[0] == 0.0
