"""Tests for change detection algorithms."""

from __future__ import annotations

import math

import pytest

from stochastic_circuit_breaker.detectors import (
    ChangeDetector,
    CUSUMDetector,
    EMADetector,
    FixedWindowDetector,
    TwoThresholdCUSUM,
)


class TestCUSUMDetector:
    def test_initial_state(self) -> None:
        d = CUSUMDetector()
        assert d.statistic == 0.0
        assert d.alarm is False

    def test_alarm_on_sustained_failures(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0)
        for _ in range(100):
            result = d.update(0.0)
            if result.alarm:
                break
        assert d.alarm is True

    def test_no_alarm_on_good_observations(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0)
        for _ in range(100):
            d.update(1.0)
        assert d.alarm is False
        # Good obs have negative LLR, so W_t stays at 0
        assert d.statistic == 0.0

    def test_reset_clears_state(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0)
        for _ in range(5):
            d.update(0.0)
        assert d.statistic > 0
        d.reset()
        assert d.statistic == 0.0
        assert d.alarm is False

    def test_bernoulli_llr_direction(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=100.0, bernoulli=True)
        # Failure (x=0): LLR = log((1-mu_1)/(1-mu_0)) > 0 → statistic increases
        result = d.update(0.0)
        assert result.statistic > 0
        # Success (x=1): LLR = log(mu_1/mu_0) < 0 → statistic stays at 0
        d.reset()
        result = d.update(1.0)
        assert result.statistic == 0.0

    def test_gaussian_mode(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0, bernoulli=False, sigma=0.2)
        for _ in range(100):
            d.update(0.3)
        assert d.alarm is True

    def test_kl_divergence_bernoulli(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0)
        kl = d.compute_kl_divergence()
        assert kl > 0
        expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
        assert abs(kl - expected) < 1e-10

    def test_kl_divergence_gaussian(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0, bernoulli=False, sigma=0.2)
        kl = d.compute_kl_divergence()
        expected = (0.9 - 0.5) ** 2 / (2 * 0.2**2)
        assert abs(kl - expected) < 1e-10

    def test_theoretical_arl1(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0)
        arl1 = d.theoretical_arl1()
        assert arl1 > 0
        assert arl1 < 100

    def test_theoretical_arl0(self) -> None:
        d = CUSUMDetector(mu_0=0.9, mu_1=0.5, h=5.0)
        arl0 = d.theoretical_arl0()
        assert arl0 > 0
        assert arl0 > d.theoretical_arl1()

    def test_invalid_parameters(self) -> None:
        with pytest.raises(ValueError, match="mu_1 < mu_0"):
            CUSUMDetector(mu_0=0.5, mu_1=0.9)
        with pytest.raises(ValueError, match="positive"):
            CUSUMDetector(h=-1)
        with pytest.raises(ValueError, match="sigma"):
            CUSUMDetector(bernoulli=False, sigma=-1)

    def test_protocol_compliance(self) -> None:
        d = CUSUMDetector()
        assert isinstance(d, ChangeDetector)


class TestTwoThresholdCUSUM:
    def test_warn_before_crit(self) -> None:
        d = TwoThresholdCUSUM(mu_0=0.9, mu_1=0.5, h_warn=3.0, h_crit=8.0)
        warn_n = -1
        crit_n = -1
        for _ in range(200):
            result = d.update(0.0)
            if d.warn_alarm and warn_n < 0:
                warn_n = result.metadata["n"]
            if d.crit_alarm and crit_n < 0:
                crit_n = result.metadata["n"]
                break
        assert warn_n > 0, "Warning alarm never fired"
        assert crit_n > 0, "Critical alarm never fired"
        assert warn_n < crit_n

    def test_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError, match="h_warn < h_crit"):
            TwoThresholdCUSUM(h_warn=8.0, h_crit=3.0)

    def test_protocol_compliance(self) -> None:
        d = TwoThresholdCUSUM()
        assert isinstance(d, ChangeDetector)


class TestEMADetector:
    def test_warmup_suppresses_alarm(self) -> None:
        d = EMADetector(alpha=0.5, threshold=0.7, warmup=10)
        for i in range(9):
            result = d.update(0.0)
            assert not result.alarm, f"Alarm during warmup at step {i}"
        result = d.update(0.0)
        assert result.alarm

    def test_no_alarm_on_good_data(self) -> None:
        d = EMADetector(alpha=0.2, threshold=0.5, warmup=5)
        for _ in range(100):
            d.update(1.0)
        assert d.alarm is False

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            EMADetector(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            EMADetector(alpha=1.0)

    def test_protocol_compliance(self) -> None:
        d = EMADetector()
        assert isinstance(d, ChangeDetector)


class TestFixedWindowDetector:
    def test_sustained_periods(self) -> None:
        d = FixedWindowDetector(window_size=5, threshold=0.7, sustained_periods=2)
        # First bad window (5 obs)
        for _ in range(5):
            d.update(0.0)
        assert not d.alarm  # Only 1 bad window
        # Second bad window
        for _ in range(5):
            d.update(0.0)
        assert d.alarm  # 2 sustained bad windows

    def test_reset_streak_on_good_window(self) -> None:
        d = FixedWindowDetector(window_size=3, threshold=0.5, sustained_periods=2)
        # One bad window
        for _ in range(3):
            d.update(0.0)
        assert not d.alarm
        # One good window breaks the streak
        for _ in range(3):
            d.update(1.0)
        assert not d.alarm
        # Another bad window: streak reset to 1
        for _ in range(3):
            d.update(0.0)
        assert not d.alarm  # Only 1 consecutive bad window

    def test_protocol_compliance(self) -> None:
        d = FixedWindowDetector()
        assert isinstance(d, ChangeDetector)
