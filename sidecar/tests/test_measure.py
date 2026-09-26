"""Unit tests for the warm/cold measurement math (no engine, any platform)."""

from cache_max.measure import Discrimination


def test_ratios_are_cold_over_warm():
    d = Discrimination(warm_ms=[100.0, 50.0], cold_ms=[200.0, 50.0])
    assert d.ratios == [2.0, 1.0]


def test_median_ratio_odd_and_even():
    assert Discrimination(warm_ms=[1, 1, 1], cold_ms=[1, 4, 9]).median_ratio == 4.0
    assert Discrimination(warm_ms=[1, 1, 1, 1], cold_ms=[1, 2, 3, 4]).median_ratio == 2.5


def test_zero_warm_is_guarded():
    d = Discrimination(warm_ms=[0.0], cold_ms=[10.0])
    assert d.ratios == [0.0]
