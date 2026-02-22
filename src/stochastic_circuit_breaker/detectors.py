"""Change detection algorithms for monitoring quality degradation.

Implements:
- CUSUM (Cumulative Sum) — optimal sequential change detection (Page 1954, Moustakides 1986)
- TwoThresholdCUSUM — CUSUM with warn + critical thresholds for 4-state breaker
- EMADetector — Exponential Moving Average baseline
- FixedWindowDetector — Sliding window mean (traditional circuit breaker approach)
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from stochastic_circuit_breaker.types import DetectorResult


@runtime_checkable
class ChangeDetector(Protocol):
    """Protocol for sequential change detection algorithms."""

    def update(self, x: float) -> DetectorResult:
        """Process a new observation and return the detection result."""
        ...

    def reset(self) -> None:
        """Reset detector state."""
        ...

    @property
    def statistic(self) -> float:
        """Current detector statistic."""
        ...

    @property
    def alarm(self) -> bool:
        """Whether the detector is currently in alarm state."""
        ...


class CUSUMDetector:
    """Cumulative Sum (CUSUM) change detector.

    Optimal sequential detector for detecting a shift from mu_0 to mu_1
    in the sense of minimizing the worst-case detection delay (Moustakides 1986).

    The CUSUM statistic accumulates:
        W_t = max(0, W_{t-1} + LLR(X_t))

    where LLR is the log-likelihood ratio between H1 (degraded) and H0 (normal).
    An alarm fires when W_t >= h.

    For Bernoulli observations (success/failure):
        LLR(1) = log(mu_1 / mu_0)          [negative: success is evidence for H0]
        LLR(0) = log((1 - mu_1) / (1 - mu_0))  [positive: failure is evidence for H1]

    For Gaussian observations:
        LLR(x) = (mu_0 - mu_1) * (x - (mu_0 + mu_1) / 2) / sigma^2

    Args:
        mu_0: Expected quality under normal operation. Range (0, 1).
        mu_1: Expected quality under degradation. Range (0, mu_0).
        h: Detection threshold. Alarm fires when statistic >= h.
        bernoulli: If True, use Bernoulli LLR. If False, use Gaussian.
        sigma: Standard deviation for Gaussian model.
    """

    def __init__(
        self,
        mu_0: float = 0.9,
        mu_1: float = 0.5,
        h: float = 5.0,
        bernoulli: bool = True,
        sigma: float = 0.1,
    ) -> None:
        if not 0 < mu_1 < mu_0 < 1:
            raise ValueError(f"Require 0 < mu_1 < mu_0 < 1, got mu_0={mu_0}, mu_1={mu_1}")
        if h <= 0:
            raise ValueError(f"Threshold h must be positive, got {h}")
        if not bernoulli and sigma <= 0:
            raise ValueError(f"sigma must be positive for Gaussian model, got {sigma}")

        self._mu_0 = mu_0
        self._mu_1 = mu_1
        self._h = h
        self._bernoulli = bernoulli
        self._sigma = sigma

        # Precompute LLR values for Bernoulli model
        if bernoulli:
            self._llr_success = math.log(mu_1 / mu_0)
            self._llr_failure = math.log((1 - mu_1) / (1 - mu_0))

        self._W: float = 0.0
        self._n: int = 0
        self._alarm: bool = False

    def _compute_llr(self, x: float) -> float:
        """Compute log-likelihood ratio log(f_1(x) / f_0(x)).

        Returns positive values when x is evidence for degradation (H1),
        negative when evidence for normal operation (H0).
        """
        if self._bernoulli:
            # For Bernoulli: interpolate for continuous values in [0, 1]
            # LLR(1) = log(mu_1/mu_0) < 0  (success → evidence for H0)
            # LLR(0) = log((1-mu_1)/(1-mu_0)) > 0  (failure → evidence for H1)
            return x * self._llr_success + (1 - x) * self._llr_failure
        else:
            # Gaussian: log(f_1(x)/f_0(x)) = (mu_1-mu_0)(x-(mu_0+mu_1)/2) / sigma^2
            midpoint = (self._mu_0 + self._mu_1) / 2
            return (self._mu_1 - self._mu_0) * (x - midpoint) / (self._sigma**2)

    def update(self, x: float) -> DetectorResult:
        """Process observation and update CUSUM statistic.

        Args:
            x: Quality observation in [0, 1].

        Returns:
            DetectorResult with current statistic and alarm state.
        """
        llr = self._compute_llr(x)
        # CUSUM recursion: W_t = max(0, W_{t-1} + log(f_1/f_0))
        # Positive LLR (degradation evidence) increases W_t toward alarm.
        self._W = max(0.0, self._W + llr)
        self._n += 1
        self._alarm = self._h <= self._W

        return DetectorResult(
            statistic=self._W,
            alarm=self._alarm,
            metadata={"llr": llr, "n": self._n},
        )

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._W = 0.0
        self._n = 0
        self._alarm = False

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic W_t."""
        return self._W

    @property
    def alarm(self) -> bool:
        """Whether W_t >= h."""
        return self._alarm

    @property
    def threshold(self) -> float:
        """Detection threshold h."""
        return self._h

    def compute_kl_divergence(self) -> float:
        """Compute KL divergence D_KL(H1 || H0).

        For Bernoulli:
            D_KL = mu_1 * log(mu_1/mu_0) + (1-mu_1) * log((1-mu_1)/(1-mu_0))

        For Gaussian:
            D_KL = (mu_0 - mu_1)^2 / (2 * sigma^2)
        """
        if self._bernoulli:
            return self._mu_1 * math.log(self._mu_1 / self._mu_0) + (1 - self._mu_1) * math.log(
                (1 - self._mu_1) / (1 - self._mu_0)
            )
        else:
            return (self._mu_0 - self._mu_1) ** 2 / (2 * self._sigma**2)

    def theoretical_arl1(self) -> float:
        """Theoretical Average Run Length under H1 (detection delay).

        Approximation: ARL_1 ~ h / D_KL(H1 || H0)
        (Wald's approximation for large h)
        """
        d_kl = self.compute_kl_divergence()
        if d_kl <= 0:
            return float("inf")
        return self._h / d_kl

    def theoretical_arl0(self) -> float:
        """Theoretical Average Run Length under H0 (false alarm rate).

        Approximation: ARL_0 ~ exp(h) / D_KL(H0 || H1)
        """
        # D_KL(H0 || H1)
        if self._bernoulli:
            d_kl_01 = self._mu_0 * math.log(self._mu_0 / self._mu_1) + (1 - self._mu_0) * math.log(
                (1 - self._mu_0) / (1 - self._mu_1)
            )
        else:
            d_kl_01 = (self._mu_0 - self._mu_1) ** 2 / (2 * self._sigma**2)

        if d_kl_01 <= 0:
            return float("inf")
        return math.exp(self._h) / d_kl_01


class TwoThresholdCUSUM:
    """CUSUM with two thresholds for the 4-state circuit breaker.

    Provides early warning (h_warn) and critical alarm (h_crit) levels.
    The circuit breaker uses these for:
        - W_t >= h_warn: CLOSED -> DEGRADED
        - W_t >= h_crit: DEGRADED -> OPEN
        - W_t == 0: DEGRADED -> CLOSED (statistic reset to zero)

    Args:
        mu_0: Expected quality under normal operation.
        mu_1: Expected quality under degradation.
        h_warn: Warning threshold (early detection).
        h_crit: Critical threshold (confirmed degradation).
        bernoulli: If True, use Bernoulli LLR.
        sigma: Standard deviation for Gaussian model.
    """

    def __init__(
        self,
        mu_0: float = 0.9,
        mu_1: float = 0.5,
        h_warn: float = 3.0,
        h_crit: float = 8.0,
        bernoulli: bool = True,
        sigma: float = 0.1,
    ) -> None:
        if h_warn >= h_crit:
            raise ValueError(f"Require h_warn < h_crit, got h_warn={h_warn}, h_crit={h_crit}")
        self._inner = CUSUMDetector(
            mu_0=mu_0, mu_1=mu_1, h=h_crit, bernoulli=bernoulli, sigma=sigma
        )
        self._h_warn = h_warn
        self._h_crit = h_crit
        self._warn_alarm = False
        self._crit_alarm = False

    def update(self, x: float) -> DetectorResult:
        """Process observation and update both threshold states."""
        result = self._inner.update(x)
        self._warn_alarm = result.statistic >= self._h_warn
        self._crit_alarm = result.statistic >= self._h_crit
        return DetectorResult(
            statistic=result.statistic,
            alarm=self._crit_alarm,
            metadata={
                **result.metadata,
                "warn_alarm": self._warn_alarm,
                "crit_alarm": self._crit_alarm,
            },
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._inner.reset()
        self._warn_alarm = False
        self._crit_alarm = False

    @property
    def statistic(self) -> float:
        return self._inner.statistic

    @property
    def alarm(self) -> bool:
        """Critical alarm (W_t >= h_crit)."""
        return self._crit_alarm

    @property
    def warn_alarm(self) -> bool:
        """Warning alarm (W_t >= h_warn)."""
        return self._warn_alarm

    @property
    def crit_alarm(self) -> bool:
        """Critical alarm (W_t >= h_crit)."""
        return self._crit_alarm

    @property
    def h_warn(self) -> float:
        return self._h_warn

    @property
    def h_crit(self) -> float:
        return self._h_crit

    def compute_kl_divergence(self) -> float:
        return self._inner.compute_kl_divergence()

    def theoretical_arl1(self) -> float:
        return self._inner.theoretical_arl1()

    def theoretical_arl0(self) -> float:
        return self._inner.theoretical_arl0()


class EMADetector:
    """Exponential Moving Average change detector.

    Baseline comparison for CUSUM. Tracks:
        EMA_t = alpha * X_t + (1 - alpha) * EMA_{t-1}

    and alarms when EMA_t <= threshold.

    Includes a warm-up period to avoid spurious alarms from insufficient data.

    Args:
        alpha: Smoothing factor in (0, 1). Higher = more responsive.
        threshold: EMA level below which alarm fires.
        warmup: Number of initial observations before alarms are enabled.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        threshold: float = 0.7,
        warmup: int = 10,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        self._alpha = alpha
        self._threshold = threshold
        self._warmup = warmup
        self._ema: float = 1.0  # Optimistic initial value
        self._n: int = 0
        self._alarm: bool = False

    def update(self, x: float) -> DetectorResult:
        """Process observation and update EMA."""
        if self._n == 0:
            self._ema = x
        else:
            self._ema = self._alpha * x + (1 - self._alpha) * self._ema
        self._n += 1

        # Only alarm after warmup
        self._alarm = self._n >= self._warmup and self._ema <= self._threshold

        return DetectorResult(
            statistic=self._ema,
            alarm=self._alarm,
            metadata={"n": self._n, "warmed_up": self._n >= self._warmup},
        )

    def reset(self) -> None:
        self._ema = 1.0
        self._n = 0
        self._alarm = False

    @property
    def statistic(self) -> float:
        return self._ema

    @property
    def alarm(self) -> bool:
        return self._alarm


class FixedWindowDetector:
    """Fixed-epoch window change detector.

    Traditional circuit breaker approach: collects non-overlapping windows of
    observations and alarms when consecutive windows have mean quality below
    the threshold.

    Args:
        window_size: Number of observations per window epoch.
        threshold: Mean quality below which a window is "bad".
        sustained_periods: Consecutive bad windows before alarm.
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.7,
        sustained_periods: int = 3,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if sustained_periods < 1:
            raise ValueError(f"sustained_periods must be >= 1, got {sustained_periods}")
        self._window_size = window_size
        self._threshold = threshold
        self._sustained_periods = sustained_periods
        self._current: list[float] = []
        self._bad_streak: int = 0
        self._alarm: bool = False
        self._mean: float = 1.0

    def update(self, x: float) -> DetectorResult:
        """Process observation and update window statistics."""
        self._current.append(x)

        if len(self._current) >= self._window_size:
            self._mean = sum(self._current) / self._window_size
            if self._mean < self._threshold:
                self._bad_streak += 1
            else:
                self._bad_streak = 0
            self._alarm = self._bad_streak >= self._sustained_periods
            self._current = []
        else:
            self._mean = sum(self._current) / len(self._current)

        return DetectorResult(
            statistic=self._mean,
            alarm=self._alarm,
            metadata={
                "window_fill": len(self._current),
                "bad_streak": self._bad_streak,
            },
        )

    def reset(self) -> None:
        self._current = []
        self._bad_streak = 0
        self._alarm = False
        self._mean = 1.0

    @property
    def statistic(self) -> float:
        return self._mean

    @property
    def alarm(self) -> bool:
        return self._alarm
