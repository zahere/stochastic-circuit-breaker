"""Stochastic circuit breaker with 4-state CUSUM-based FSM.

State machine:
    CLOSED ──[W_t >= h_warn]──> DEGRADED
    DEGRADED ──[W_t >= h_crit]──> OPEN
    DEGRADED ──[W_t == 0]──> CLOSED
    OPEN ──[timeout]──> PROBING
    PROBING ──[t-test pass]──> CLOSED
    PROBING ──[t-test fail]──> OPEN

Thread-safe: a lock protects state mutations, but the wrapped function
executes outside the lock to avoid blocking concurrent callers.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from stochastic_circuit_breaker.detectors import TwoThresholdCUSUM
from stochastic_circuit_breaker.quality import BinaryQuality, QualityFunction
from stochastic_circuit_breaker.types import (
    BreakerConfig,
    BreakerSnapshot,
    BreakerState,
    CallResult,
    StateTransition,
    TransitionEvent,
)

T = TypeVar("T")


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the breaker is OPEN.

    Attributes:
        state: Current breaker state.
        statistic: Current CUSUM statistic.
        time_until_probe: Seconds until the breaker transitions to PROBING.
    """

    def __init__(self, state: BreakerState, statistic: float, time_until_probe: float) -> None:
        self.state = state
        self.statistic = statistic
        self.time_until_probe = max(0.0, time_until_probe)
        super().__init__(
            f"Circuit is {state.value} (W_t={statistic:.2f}, "
            f"probing in {self.time_until_probe:.1f}s)"
        )


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via Lentz's continued fraction.

    Used for computing p-values of the t-distribution without scipy.
    Implements the continued fraction expansion from Numerical Recipes (section 6.4).
    """
    if x < 0 or x > 1:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use the symmetry relation when x > (a+1)/(a+b+2) for convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(b, a, 1.0 - x)

    # Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    log_prefix = (
        a * math.log(x)
        + b * math.log(1 - x)
        - math.log(a)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )

    # Lentz's continued fraction method
    tiny = 1e-30
    eps = 1e-14
    max_iter = 200

    # f = 1 + d_1 / (1 + d_2 / (1 + ...))
    # Using modified Lentz method
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, max_iter + 1):
        # Even step: d_{2m}
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        # Odd step: d_{2m+1}
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta

        if abs(delta - 1.0) < eps:
            break

    return math.exp(log_prefix) * result


def _t_cdf(t_stat: float, df: int) -> float:
    """CDF of Student's t-distribution at t_stat with df degrees of freedom.

    Uses the relationship: CDF(t) = 1 - 0.5 * I_{v/(v+t^2)}(v/2, 1/2)
    where v = df, I is the regularized incomplete beta function.
    """
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")
    x = df / (df + t_stat * t_stat)
    beta_val = _regularized_incomplete_beta(df / 2, 0.5, x)
    if t_stat >= 0:
        return 1.0 - 0.5 * beta_val
    else:
        return 0.5 * beta_val


def _one_sample_t_test_pvalue(samples: list[float], mu_0: float) -> float:
    """One-sample t-test: H0: mu <= mu_0 vs H1: mu > mu_0.

    Returns the one-sided p-value.
    """
    n = len(samples)
    if n < 2:
        return 1.0  # Cannot compute, assume no evidence

    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
    if variance <= 0:
        # All samples identical
        return 0.0 if mean > mu_0 else 1.0

    se = math.sqrt(variance / n)
    t_stat = (mean - mu_0) / se
    df = n - 1
    # P-value for one-sided test (upper tail)
    return 1.0 - _t_cdf(t_stat, df)


class CircuitBreaker:
    """Stochastic circuit breaker with 4-state CUSUM-based detection.

    Wraps function calls and monitors output quality using sequential change
    detection. Unlike deterministic circuit breakers that count failures,
    this uses the CUSUM algorithm to detect statistically significant
    degradation in a stochastic system.

    Args:
        quality_fn: Function to score outputs. Defaults to BinaryQuality.
        config: Breaker configuration. Defaults to BreakerConfig().
        detector: Custom change detector. Defaults to TwoThresholdCUSUM from config.
        on_transition: Optional callback invoked on state transitions.
    """

    def __init__(
        self,
        quality_fn: QualityFunction | None = None,
        config: BreakerConfig | None = None,
        detector: TwoThresholdCUSUM | None = None,
        on_transition: Callable[[StateTransition], Any] | None = None,
    ) -> None:
        self._config = config or BreakerConfig()
        self._quality_fn = quality_fn or BinaryQuality()
        self._detector = detector or TwoThresholdCUSUM(
            mu_0=self._config.mu_0,
            mu_1=self._config.mu_1,
            h_warn=self._config.h_warn,
            h_crit=self._config.h_crit,
            bernoulli=self._config.bernoulli,
            sigma=self._config.sigma,
        )
        self._on_transition = on_transition

        self._state = BreakerState.CLOSED
        self._lock = threading.Lock()
        self._call_count = 0
        self._failure_count = 0
        self._last_transition: StateTransition | None = None
        self._open_since: float | None = None
        self._probe_results: list[float] = []

    @property
    def state(self) -> BreakerState:
        """Current breaker state."""
        return self._state

    @property
    def statistic(self) -> float:
        """Current detector statistic."""
        return self._detector.statistic

    @property
    def config(self) -> BreakerConfig:
        """Breaker configuration."""
        return self._config

    def _transition(self, to_state: BreakerState, event: TransitionEvent) -> StateTransition:
        """Record and execute a state transition. Caller must hold the lock."""
        transition = StateTransition(
            from_state=self._state,
            to_state=to_state,
            event=event,
            timestamp=time.monotonic(),
            statistic=self._detector.statistic,
        )
        self._state = to_state
        self._last_transition = transition

        if to_state == BreakerState.OPEN:
            self._open_since = time.monotonic()
            self._probe_results = []
        elif to_state == BreakerState.PROBING:
            self._probe_results = []
        elif to_state == BreakerState.CLOSED:
            self._open_since = None
            self._probe_results = []
            self._detector.reset()

        if self._on_transition is not None:
            with contextlib.suppress(Exception):
                self._on_transition(transition)

        return transition

    def _check_open_timeout(self) -> StateTransition | None:
        """Check if OPEN timeout has elapsed. Caller must hold the lock."""
        if (
            self._state == BreakerState.OPEN
            and self._open_since is not None
            and (time.monotonic() - self._open_since) >= self._config.open_timeout
        ):
            return self._transition(BreakerState.PROBING, TransitionEvent.OPEN_TO_PROBING)
        return None

    def _process_observation(self, quality_score: float) -> StateTransition | None:
        """Process a quality observation and handle state transitions. Caller must hold lock."""
        result = self._detector.update(quality_score)

        if self._state == BreakerState.CLOSED:
            if result.metadata.get("warn_alarm", False):
                return self._transition(BreakerState.DEGRADED, TransitionEvent.CLOSED_TO_DEGRADED)

        elif self._state == BreakerState.DEGRADED:
            if result.metadata.get("crit_alarm", False):
                return self._transition(BreakerState.OPEN, TransitionEvent.DEGRADED_TO_OPEN)
            elif result.statistic == 0.0:
                return self._transition(BreakerState.CLOSED, TransitionEvent.DEGRADED_TO_CLOSED)

        elif self._state == BreakerState.PROBING:
            self._probe_results.append(quality_score)
            if len(self._probe_results) >= self._config.probe_window:
                p_value = _one_sample_t_test_pvalue(self._probe_results, self._config.mu_1)
                if p_value < self._config.recovery_alpha:
                    return self._transition(BreakerState.CLOSED, TransitionEvent.PROBING_TO_CLOSED)
                else:
                    return self._transition(BreakerState.OPEN, TransitionEvent.PROBING_TO_OPEN)

        return None

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> CallResult:
        """Execute a function through the circuit breaker.

        Args:
            fn: The function to call.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            CallResult with the output, quality score, and breaker state.

        Raises:
            CircuitOpenError: If the breaker is OPEN and timeout hasn't elapsed.
        """
        transition: StateTransition | None = None

        # Pre-call state check (under lock)
        with self._lock:
            # Check OPEN timeout first
            timeout_transition = self._check_open_timeout()
            if timeout_transition is not None:
                transition = timeout_transition

            if self._state == BreakerState.OPEN:
                time_until_probe = 0.0
                if self._open_since is not None:
                    elapsed = time.monotonic() - self._open_since
                    time_until_probe = self._config.open_timeout - elapsed
                raise CircuitOpenError(self._state, self._detector.statistic, time_until_probe)

        # Execute the wrapped function OUTSIDE the lock
        start = time.monotonic()
        output: Any = None
        quality_score: float = 0.0
        exception: BaseException | None = None

        try:
            output = fn(*args, **kwargs)
        except BaseException as e:
            exception = e
            quality_score = 0.0
        else:
            try:
                quality_score = self._quality_fn(output)
            except Exception:
                quality_score = 0.0
            # Clamp to [0, 1]
            quality_score = max(0.0, min(1.0, quality_score))

        elapsed = time.monotonic() - start

        # Post-call state update (under lock)
        with self._lock:
            self._call_count += 1
            if quality_score < 0.5:
                self._failure_count += 1

            obs_transition = self._process_observation(quality_score)
            if obs_transition is not None:
                transition = obs_transition

        # Re-raise original exception after recording
        if exception is not None:
            raise exception

        return CallResult(
            output=output,
            quality_score=quality_score,
            state=self._state,
            statistic=self._detector.statistic,
            elapsed=elapsed,
            transition=transition,
        )

    async def acall(
        self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> CallResult:
        """Execute an async function through the circuit breaker.

        Args:
            fn: The async function to call.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            CallResult with the output, quality score, and breaker state.

        Raises:
            CircuitOpenError: If the breaker is OPEN and timeout hasn't elapsed.
        """
        transition: StateTransition | None = None

        with self._lock:
            timeout_transition = self._check_open_timeout()
            if timeout_transition is not None:
                transition = timeout_transition

            if self._state == BreakerState.OPEN:
                time_until_probe = 0.0
                if self._open_since is not None:
                    elapsed = time.monotonic() - self._open_since
                    time_until_probe = self._config.open_timeout - elapsed
                raise CircuitOpenError(self._state, self._detector.statistic, time_until_probe)

        start = time.monotonic()
        output: Any = None
        quality_score: float = 0.0
        exception: BaseException | None = None

        try:
            output = await fn(*args, **kwargs)
        except BaseException as e:
            exception = e
            quality_score = 0.0
        else:
            try:
                quality_score = self._quality_fn(output)
            except Exception:
                quality_score = 0.0
            quality_score = max(0.0, min(1.0, quality_score))

        elapsed = time.monotonic() - start

        with self._lock:
            self._call_count += 1
            if quality_score < 0.5:
                self._failure_count += 1

            obs_transition = self._process_observation(quality_score)
            if obs_transition is not None:
                transition = obs_transition

        if exception is not None:
            raise exception

        return CallResult(
            output=output,
            quality_score=quality_score,
            state=self._state,
            statistic=self._detector.statistic,
            elapsed=elapsed,
            transition=transition,
        )

    def trip(self) -> StateTransition:
        """Manually force the breaker to OPEN state."""
        with self._lock:
            return self._transition(BreakerState.OPEN, TransitionEvent.MANUAL_TRIP)

    def reset(self) -> StateTransition:
        """Manually force the breaker to CLOSED state."""
        with self._lock:
            return self._transition(BreakerState.CLOSED, TransitionEvent.MANUAL_RESET)

    def snapshot(self) -> BreakerSnapshot:
        """Return an immutable snapshot of the current breaker state."""
        with self._lock:
            return BreakerSnapshot(
                state=self._state,
                statistic=self._detector.statistic,
                call_count=self._call_count,
                failure_count=self._failure_count,
                last_transition=self._last_transition,
                open_since=self._open_since,
                probe_results=tuple(self._probe_results),
                created_at=time.monotonic(),
            )

    def __enter__(self) -> CircuitBreaker:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    async def __aenter__(self) -> CircuitBreaker:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass
