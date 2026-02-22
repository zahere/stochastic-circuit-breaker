"""Stochastic Circuit Breaker — statistically optimal fault detection for AI agents.

A circuit breaker designed for stochastic systems like LLM agents, where retrying
is resampling, not replaying. Uses CUSUM sequential change detection (Moustakides 1986)
for provably optimal degradation detection.

Quick start::

    from stochastic_circuit_breaker import CircuitBreaker

    cb = CircuitBreaker()
    result = cb.call(my_llm_function, prompt="Hello")
    print(result.state, result.quality_score)
"""

from stochastic_circuit_breaker.breaker import CircuitBreaker, CircuitOpenError
from stochastic_circuit_breaker.detectors import (
    ChangeDetector,
    CUSUMDetector,
    EMADetector,
    FixedWindowDetector,
    TwoThresholdCUSUM,
)
from stochastic_circuit_breaker.quality import (
    BinaryQuality,
    ContinuousQuality,
    QualityFunction,
    ThresholdQuality,
)
from stochastic_circuit_breaker.registry import BreakerRegistry
from stochastic_circuit_breaker.types import (
    BreakerConfig,
    BreakerSnapshot,
    BreakerState,
    CallResult,
    DetectorResult,
    StateTransition,
    TransitionEvent,
)

__all__ = [
    # Core
    "CircuitBreaker",
    "CircuitOpenError",
    "BreakerRegistry",
    # Configuration
    "BreakerConfig",
    "BreakerState",
    "TransitionEvent",
    # Results
    "CallResult",
    "BreakerSnapshot",
    "StateTransition",
    "DetectorResult",
    # Quality functions
    "QualityFunction",
    "BinaryQuality",
    "ThresholdQuality",
    "ContinuousQuality",
    # Detectors
    "ChangeDetector",
    "CUSUMDetector",
    "TwoThresholdCUSUM",
    "EMADetector",
    "FixedWindowDetector",
]

__version__ = "0.1.0"
