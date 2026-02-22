"""Core types for the stochastic circuit breaker.

Enums, frozen dataclasses for immutable state, and mutable configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BreakerState(str, Enum):
    """Circuit breaker states.

    4-state automaton:
        CLOSED -> DEGRADED -> OPEN -> PROBING -> CLOSED
    """

    CLOSED = "closed"
    DEGRADED = "degraded"
    OPEN = "open"
    PROBING = "probing"


class TransitionEvent(str, Enum):
    """Events that trigger state transitions."""

    CLOSED_TO_DEGRADED = "closed_to_degraded"
    DEGRADED_TO_OPEN = "degraded_to_open"
    DEGRADED_TO_CLOSED = "degraded_to_closed"
    OPEN_TO_PROBING = "open_to_probing"
    PROBING_TO_CLOSED = "probing_to_closed"
    PROBING_TO_OPEN = "probing_to_open"
    MANUAL_TRIP = "manual_trip"
    MANUAL_RESET = "manual_reset"


@dataclass(frozen=True)
class StateTransition:
    """Immutable record of a state transition."""

    from_state: BreakerState
    to_state: BreakerState
    event: TransitionEvent
    timestamp: float
    statistic: float


@dataclass(frozen=True)
class CallResult:
    """Immutable result of a wrapped call."""

    output: object
    quality_score: float
    state: BreakerState
    statistic: float
    elapsed: float
    transition: StateTransition | None = None


@dataclass(frozen=True)
class BreakerSnapshot:
    """Immutable snapshot of breaker state for inspection."""

    state: BreakerState
    statistic: float
    call_count: int
    failure_count: int
    last_transition: StateTransition | None
    open_since: float | None
    probe_results: tuple[float, ...]
    created_at: float


@dataclass
class BreakerConfig:
    """Configuration for the stochastic circuit breaker.

    Attributes:
        mu_0: Expected quality under normal operation (H0). Range (0, 1).
        mu_1: Expected quality under degradation (H1). Range (0, mu_0).
        h_warn: CUSUM threshold for CLOSED -> DEGRADED transition.
        h_crit: CUSUM threshold for DEGRADED -> OPEN transition.
        probe_window: Number of observations to collect in PROBING state.
        recovery_alpha: Significance level for recovery t-test.
        open_timeout: Seconds to wait in OPEN before transitioning to PROBING.
        bernoulli: If True, use Bernoulli LLR. If False, use Gaussian LLR.
        sigma: Standard deviation for Gaussian model (ignored if bernoulli=True).
    """

    mu_0: float = 0.9
    mu_1: float = 0.5
    h_warn: float = 3.0
    h_crit: float = 8.0
    probe_window: int = 10
    recovery_alpha: float = 0.05
    open_timeout: float = 30.0
    bernoulli: bool = True
    sigma: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.mu_1 < self.mu_0 < 1:
            raise ValueError(
                f"Require 0 < mu_1 < mu_0 < 1, got mu_0={self.mu_0}, mu_1={self.mu_1}"
            )
        if self.h_warn <= 0 or self.h_crit <= 0:
            raise ValueError(
                f"Thresholds must be positive, got h_warn={self.h_warn}, h_crit={self.h_crit}"
            )
        if self.h_warn >= self.h_crit:
            raise ValueError(
                f"Require h_warn < h_crit, got h_warn={self.h_warn}, h_crit={self.h_crit}"
            )
        if self.probe_window < 2:
            raise ValueError(f"probe_window must be >= 2, got {self.probe_window}")
        if not 0 < self.recovery_alpha < 1:
            raise ValueError(f"recovery_alpha must be in (0, 1), got {self.recovery_alpha}")
        if self.open_timeout < 0:
            raise ValueError(f"open_timeout must be >= 0, got {self.open_timeout}")
        if not self.bernoulli and self.sigma <= 0:
            raise ValueError(f"sigma must be positive for Gaussian model, got {self.sigma}")


@dataclass
class DetectorResult:
    """Result from a change detector update."""

    statistic: float
    alarm: bool
    metadata: dict[str, object] = field(default_factory=dict)
