"""Basic usage: Wrap a function and observe state transitions."""

import random

from stochastic_circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    ContinuousQuality,
    StateTransition,
)


def on_transition(t: StateTransition) -> None:
    print(f"  [TRANSITION] {t.from_state.value} -> {t.to_state.value} (W_t={t.statistic:.2f})")


def unreliable_service(failure_rate: float = 0.1) -> dict[str, float]:
    """Simulate a service that sometimes returns low-quality responses."""
    if random.random() < failure_rate:
        return {"confidence": random.uniform(0.1, 0.3)}
    return {"confidence": random.uniform(0.7, 0.99)}


def main() -> None:
    cb = CircuitBreaker(
        quality_fn=ContinuousQuality(key=lambda r: r["confidence"]),
        config=BreakerConfig(
            mu_0=0.8,  # Expected quality under normal operation
            mu_1=0.4,  # Expected quality when degraded
            h_warn=3.0,  # Warning threshold
            h_crit=6.0,  # Critical threshold
            open_timeout=2.0,  # Seconds before probing
        ),
        on_transition=on_transition,
    )

    print("=== Normal operation (10% failure rate) ===")
    for i in range(30):
        try:
            result = cb.call(unreliable_service, failure_rate=0.1)
            print(f"  Call {i}: quality={result.quality_score:.2f}, state={result.state.value}")
        except CircuitOpenError as e:
            print(f"  Call {i}: BLOCKED - {e}")

    print("\n=== Degraded service (70% failure rate) ===")
    for i in range(30, 60):
        try:
            result = cb.call(unreliable_service, failure_rate=0.7)
            print(f"  Call {i}: quality={result.quality_score:.2f}, state={result.state.value}")
        except CircuitOpenError as e:
            print(f"  Call {i}: BLOCKED - {e}")

    snap = cb.snapshot()
    print("\n=== Summary ===")
    print(f"  State: {snap.state.value}")
    print(f"  CUSUM statistic: {snap.statistic:.2f}")
    print(f"  Total calls: {snap.call_count}")
    print(f"  Failures: {snap.failure_count}")


if __name__ == "__main__":
    main()
