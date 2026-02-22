"""Tests for the BreakerRegistry."""

from __future__ import annotations

import gc
import threading

from stochastic_circuit_breaker.breaker import CircuitBreaker
from stochastic_circuit_breaker.registry import BreakerRegistry
from stochastic_circuit_breaker.types import BreakerState


class TestRegistration:
    def test_register_and_get(self) -> None:
        registry = BreakerRegistry()
        cb = CircuitBreaker()
        registry.register("agent-1", cb)
        assert registry.get("agent-1") is cb

    def test_get_nonexistent(self) -> None:
        registry = BreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_unregister(self) -> None:
        registry = BreakerRegistry()
        cb = CircuitBreaker()
        registry.register("agent-1", cb)
        assert registry.unregister("agent-1") is True
        assert registry.get("agent-1") is None
        assert registry.unregister("agent-1") is False

    def test_replace_registration(self) -> None:
        registry = BreakerRegistry()
        cb1 = CircuitBreaker()
        cb2 = CircuitBreaker()
        registry.register("agent-1", cb1)
        registry.register("agent-1", cb2)
        assert registry.get("agent-1") is cb2


class TestWeakRef:
    def test_weakref_cleanup(self) -> None:
        registry = BreakerRegistry()
        cb = CircuitBreaker()
        registry.register("agent-1", cb)
        assert "agent-1" in registry

        # Drop reference and force GC
        del cb
        gc.collect()

        # Should be cleaned up
        assert registry.get("agent-1") is None
        assert len(registry) == 0

    def test_multiple_weakrefs(self) -> None:
        registry = BreakerRegistry()
        alive: dict[str, CircuitBreaker] = {}
        for i in range(5):
            cb = CircuitBreaker()
            registry.register(f"agent-{i}", cb)
            if i % 2 == 0:
                alive[f"agent-{i}"] = cb
        # After loop, only `alive` holds refs for agent-0, agent-2, agent-4
        del cb  # Drop last loop variable
        gc.collect()

        names = set(registry.names())
        assert names == {"agent-0", "agent-2", "agent-4"}
        assert len(registry) == 3


class TestFleetInspection:
    def test_snapshots(self) -> None:
        registry = BreakerRegistry()
        cb1 = CircuitBreaker()
        cb2 = CircuitBreaker()
        registry.register("a", cb1)
        registry.register("b", cb2)
        snaps = registry.snapshots()
        assert len(snaps) == 2
        assert snaps["a"].state == BreakerState.CLOSED
        assert snaps["b"].state == BreakerState.CLOSED

    def test_open_breakers(self) -> None:
        registry = BreakerRegistry()
        cb1 = CircuitBreaker()
        cb2 = CircuitBreaker()
        cb3 = CircuitBreaker()
        registry.register("a", cb1)
        registry.register("b", cb2)
        registry.register("c", cb3)

        cb2.trip()
        assert registry.open_breakers() == ["b"]

    def test_degraded_breakers(self) -> None:
        from stochastic_circuit_breaker.quality import ContinuousQuality
        from stochastic_circuit_breaker.types import BreakerConfig

        registry = BreakerRegistry()
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=lambda _: 0.0),
            config=BreakerConfig(mu_0=0.9, mu_1=0.5, h_warn=2.0, h_crit=50.0),
        )
        registry.register("agent", cb)

        # Push into DEGRADED
        for _ in range(20):
            cb.call(lambda: "bad")
        assert cb.state == BreakerState.DEGRADED
        assert registry.degraded_breakers() == ["agent"]

    def test_summary(self) -> None:
        registry = BreakerRegistry()
        cb1 = CircuitBreaker()
        cb2 = CircuitBreaker()
        registry.register("a", cb1)
        registry.register("b", cb2)
        cb2.trip()

        summary = registry.summary()
        assert summary["closed"] == 1
        assert summary["open"] == 1
        assert summary["degraded"] == 0
        assert summary["probing"] == 0

    def test_contains(self) -> None:
        registry = BreakerRegistry()
        cb = CircuitBreaker()
        registry.register("x", cb)
        assert "x" in registry
        assert "y" not in registry


class TestThreadSafety:
    def test_concurrent_registration(self) -> None:
        registry = BreakerRegistry()
        errors: list[Exception] = []
        breakers: list[CircuitBreaker] = []

        def worker(n: int) -> None:
            try:
                cb = CircuitBreaker()
                breakers.append(cb)  # Keep reference alive
                registry.register(f"agent-{n}", cb)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry) == 20
