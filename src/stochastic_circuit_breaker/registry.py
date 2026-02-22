"""Multi-agent circuit breaker registry with weak references.

Manages a fleet of circuit breakers for monitoring multiple agents/services.
Uses weak references for automatic cleanup when breakers are garbage collected.
"""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, Any

from stochastic_circuit_breaker.types import BreakerSnapshot, BreakerState

if TYPE_CHECKING:
    from stochastic_circuit_breaker.breaker import CircuitBreaker


class BreakerRegistry:
    """Thread-safe registry for managing multiple circuit breakers.

    Uses weak references so breakers are automatically removed when the
    owning code drops its reference. This prevents memory leaks in
    long-running agent systems where agents are dynamically created/destroyed.

    Example::

        registry = BreakerRegistry()
        cb = CircuitBreaker()
        registry.register("agent-1", cb)

        # Inspect fleet
        print(registry.summary())
        for name in registry.open_breakers():
            print(f"ALERT: {name} is OPEN")
    """

    def __init__(self) -> None:
        self._breakers: dict[str, weakref.ref[CircuitBreaker]] = {}
        self._lock = threading.Lock()

    def _make_cleanup(self, name: str) -> Any:
        """Create a weak reference callback that removes the entry on GC."""

        def _cleanup(ref: weakref.ref[CircuitBreaker]) -> None:
            with self._lock:
                # Only remove if the ref is still the one we registered
                if self._breakers.get(name) is ref:
                    del self._breakers[name]

        return _cleanup

    def register(self, name: str, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker under a name.

        If a breaker is already registered with this name, it is replaced.

        Args:
            name: Unique identifier for this breaker.
            breaker: The CircuitBreaker instance to register.
        """
        with self._lock:
            self._breakers[name] = weakref.ref(breaker, self._make_cleanup(name))

    def unregister(self, name: str) -> bool:
        """Remove a breaker from the registry.

        Args:
            name: The breaker name to remove.

        Returns:
            True if the breaker was found and removed, False otherwise.
        """
        with self._lock:
            return self._breakers.pop(name, None) is not None

    def get(self, name: str) -> CircuitBreaker | None:
        """Retrieve a circuit breaker by name.

        Returns None if the breaker was never registered or has been
        garbage collected.
        """
        with self._lock:
            ref = self._breakers.get(name)
        if ref is None:
            return None
        breaker = ref()
        if breaker is None:
            # Already GC'd, clean up stale entry
            with self._lock:
                if self._breakers.get(name) is ref:
                    del self._breakers[name]
            return None
        return breaker

    def names(self) -> list[str]:
        """Return names of all live (non-GC'd) breakers."""
        result: list[str] = []
        stale: list[str] = []
        with self._lock:
            for name, ref in self._breakers.items():
                if ref() is not None:
                    result.append(name)
                else:
                    stale.append(name)
            for name in stale:
                del self._breakers[name]
        return result

    def snapshots(self) -> dict[str, BreakerSnapshot]:
        """Return snapshots for all live breakers."""
        result: dict[str, BreakerSnapshot] = {}
        for name in self.names():
            breaker = self.get(name)
            if breaker is not None:
                result[name] = breaker.snapshot()
        return result

    def open_breakers(self) -> list[str]:
        """Return names of breakers currently in OPEN state."""
        return [name for name, snap in self.snapshots().items() if snap.state == BreakerState.OPEN]

    def degraded_breakers(self) -> list[str]:
        """Return names of breakers currently in DEGRADED state."""
        return [
            name for name, snap in self.snapshots().items() if snap.state == BreakerState.DEGRADED
        ]

    def summary(self) -> dict[str, int]:
        """Return count of breakers per state.

        Example::

            {"closed": 5, "degraded": 1, "open": 0, "probing": 0}
        """
        counts: dict[str, int] = {s.value: 0 for s in BreakerState}
        for snap in self.snapshots().values():
            counts[snap.state.value] += 1
        return counts

    def __len__(self) -> int:
        return len(self.names())

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None
