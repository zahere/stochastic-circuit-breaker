"""Multi-agent monitoring with BreakerRegistry."""

import contextlib
import random

from stochastic_circuit_breaker import (
    BreakerConfig,
    BreakerRegistry,
    CircuitBreaker,
    CircuitOpenError,
    ContinuousQuality,
)

AGENTS = [
    ("detection-agent", 0.05),  # Very reliable
    ("diagnosis-agent", 0.15),  # Slightly degraded
    ("remediation-agent", 0.5),  # Heavily degraded
    ("communication-agent", 0.1),
    ("documentation-agent", 0.08),
]


def simulate_agent(failure_rate: float) -> dict[str, float]:
    if random.random() < failure_rate:
        return {"quality": random.uniform(0.0, 0.3)}
    return {"quality": random.uniform(0.7, 1.0)}


def main() -> None:
    registry = BreakerRegistry()
    breakers: dict[str, CircuitBreaker] = {}

    config = BreakerConfig(
        mu_0=0.8,
        mu_1=0.4,
        h_warn=3.0,
        h_crit=6.0,
        open_timeout=5.0,
    )

    # Create and register breakers for each agent
    for name, _ in AGENTS:
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=lambda r: r["quality"]),
            config=config,
        )
        registry.register(name, cb)
        breakers[name] = cb

    # Simulate workload
    print("=== Simulating 50 calls per agent ===\n")
    for _ in range(50):
        for name, failure_rate in AGENTS:
            cb = breakers[name]
            with contextlib.suppress(CircuitOpenError):
                cb.call(simulate_agent, failure_rate)

    # Fleet inspection
    print("=== Fleet Status ===")
    summary = registry.summary()
    for state, count in summary.items():
        print(f"  {state}: {count}")

    open_agents = registry.open_breakers()
    if open_agents:
        print(f"\n  ALERT: {len(open_agents)} agent(s) OPEN: {', '.join(open_agents)}")

    degraded_agents = registry.degraded_breakers()
    if degraded_agents:
        print(f"  WARNING: {len(degraded_agents)} agent(s) DEGRADED: {', '.join(degraded_agents)}")

    # Detailed snapshots
    print("\n=== Detailed Snapshots ===")
    for name, snap in registry.snapshots().items():
        print(
            f"  {name}: state={snap.state.value}, "
            f"W_t={snap.statistic:.2f}, "
            f"calls={snap.call_count}, "
            f"failures={snap.failure_count}"
        )


if __name__ == "__main__":
    main()
