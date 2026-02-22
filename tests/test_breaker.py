"""Tests for the stochastic circuit breaker."""

from __future__ import annotations

import contextlib
import threading
import time

import pytest

from stochastic_circuit_breaker.breaker import (
    CircuitBreaker,
    CircuitOpenError,
    _one_sample_t_test_pvalue,
    _regularized_incomplete_beta,
    _t_cdf,
)
from stochastic_circuit_breaker.quality import ContinuousQuality
from stochastic_circuit_breaker.types import (
    BreakerConfig,
    BreakerState,
    TransitionEvent,
)


def _always_zero(_: object) -> float:
    return 0.0


def _always_one(_: object) -> float:
    return 1.0


def _low_quality(_: object) -> float:
    return 0.1


def _high_quality(_: object) -> float:
    return 0.95


class TestStateTransitions:
    """Test all 6 state transitions + manual trip/reset."""

    def test_closed_to_degraded(self) -> None:
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=_always_zero),
            config=BreakerConfig(mu_0=0.9, mu_1=0.5, h_warn=2.0, h_crit=50.0),
        )
        for _ in range(20):
            cb.call(lambda: "bad")
        assert cb.state == BreakerState.DEGRADED

    def test_degraded_to_open(self) -> None:
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=_always_zero),
            config=BreakerConfig(mu_0=0.9, mu_1=0.5, h_warn=2.0, h_crit=5.0),
        )
        for _ in range(50):
            try:
                cb.call(lambda: "bad")
            except CircuitOpenError:
                break
        assert cb.state == BreakerState.OPEN

    def test_degraded_to_closed(self) -> None:
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=lambda x: 0.0 if x == "bad" else 1.0),
            config=BreakerConfig(mu_0=0.9, mu_1=0.5, h_warn=1.0, h_crit=50.0),
        )
        # Push into DEGRADED
        for _ in range(10):
            cb.call(lambda: "bad")
        assert cb.state == BreakerState.DEGRADED
        # Recover with good observations
        for _ in range(200):
            cb.call(lambda: "good")
            if cb.state == BreakerState.CLOSED:
                break
        assert cb.state == BreakerState.CLOSED

    def test_open_to_probing(self) -> None:
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=_high_quality),
            config=BreakerConfig(mu_0=0.9, mu_1=0.5, h_warn=2.0, h_crit=5.0, open_timeout=0.01),
        )
        cb.trip()
        assert cb.state == BreakerState.OPEN
        time.sleep(0.02)
        # Next call triggers timeout check -> PROBING, then processes observation
        with contextlib.suppress(CircuitOpenError):
            cb.call(lambda: "ok")
        assert cb.state in (BreakerState.PROBING, BreakerState.CLOSED)

    def test_probing_to_closed_on_good_quality(self) -> None:
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=_high_quality),
            config=BreakerConfig(
                mu_0=0.9,
                mu_1=0.5,
                h_warn=2.0,
                h_crit=5.0,
                open_timeout=0.01,
                probe_window=3,
                recovery_alpha=0.5,
            ),
        )
        cb.trip()
        time.sleep(0.02)
        for _ in range(10):
            try:
                cb.call(lambda: "good")
            except CircuitOpenError:
                time.sleep(0.02)
        assert cb.state == BreakerState.CLOSED

    def test_probing_to_open_on_bad_quality(self) -> None:
        transitions: list[object] = []
        cb = CircuitBreaker(
            quality_fn=ContinuousQuality(key=_low_quality),
            config=BreakerConfig(
                mu_0=0.9,
                mu_1=0.5,
                h_warn=2.0,
                h_crit=5.0,
                open_timeout=0.01,
                probe_window=3,
                recovery_alpha=0.05,
            ),
            on_transition=transitions.append,
        )
        cb.trip()
        time.sleep(0.02)
        for _ in range(20):
            try:
                cb.call(lambda: "bad")
            except CircuitOpenError:
                time.sleep(0.02)
        # Verify PROBING -> OPEN transition occurred
        events = [t.event for t in transitions]  # type: ignore[union-attr]
        assert TransitionEvent.PROBING_TO_OPEN in events

    def test_manual_trip(self) -> None:
        cb = CircuitBreaker()
        transition = cb.trip()
        assert cb.state == BreakerState.OPEN
        assert transition.event == TransitionEvent.MANUAL_TRIP

    def test_manual_reset(self) -> None:
        cb = CircuitBreaker()
        cb.trip()
        transition = cb.reset()
        assert cb.state == BreakerState.CLOSED
        assert transition.event == TransitionEvent.MANUAL_RESET


class TestCircuitOpenError:
    def test_raised_when_open(self) -> None:
        cb = CircuitBreaker()
        cb.trip()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: "hello")
        assert exc_info.value.state == BreakerState.OPEN
        assert exc_info.value.statistic >= 0
        assert exc_info.value.time_until_probe >= 0

    def test_attributes(self) -> None:
        err = CircuitOpenError(BreakerState.OPEN, 12.5, 25.0)
        assert err.state == BreakerState.OPEN
        assert err.statistic == 12.5
        assert err.time_until_probe == 25.0

    def test_negative_time_clamped(self) -> None:
        err = CircuitOpenError(BreakerState.OPEN, 0.0, -5.0)
        assert err.time_until_probe == 0.0


class TestExceptionHandling:
    def test_wrapped_fn_exception_is_reraised(self) -> None:
        cb = CircuitBreaker()

        def failing() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            cb.call(failing)

    def test_wrapped_fn_exception_records_failure(self) -> None:
        cb = CircuitBreaker()

        def failing() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            cb.call(failing)
        snap = cb.snapshot()
        assert snap.call_count == 1
        assert snap.failure_count == 1

    def test_quality_fn_exception_treated_as_zero(self) -> None:
        def bad_quality(output: object) -> float:
            raise TypeError("broken quality fn")

        cb = CircuitBreaker(quality_fn=bad_quality)  # type: ignore[arg-type]
        result = cb.call(lambda: "hello")
        assert result.quality_score == 0.0

    def test_quality_score_clamped(self) -> None:
        cb = CircuitBreaker(quality_fn=ContinuousQuality(key=lambda _: 1.5))
        result = cb.call(lambda: "ok")
        assert result.quality_score == 1.0

        cb2 = CircuitBreaker(quality_fn=ContinuousQuality(key=lambda _: -0.5))
        result2 = cb2.call(lambda: "ok")
        assert result2.quality_score == 0.0


class TestAsyncSupport:
    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        cb = CircuitBreaker()

        async def my_fn() -> str:
            return "async result"

        result = await cb.acall(my_fn)
        assert result.output == "async result"
        assert result.quality_score == 1.0

    @pytest.mark.asyncio
    async def test_async_exception_reraised(self) -> None:
        cb = CircuitBreaker()

        async def failing() -> None:
            raise ValueError("async fail")

        with pytest.raises(ValueError, match="async fail"):
            await cb.acall(failing)

    @pytest.mark.asyncio
    async def test_async_circuit_open(self) -> None:
        cb = CircuitBreaker()
        cb.trip()

        async def noop() -> str:
            return "nope"

        with pytest.raises(CircuitOpenError):
            await cb.acall(noop)


class TestContextManager:
    def test_sync_context_manager(self) -> None:
        with CircuitBreaker() as cb:
            result = cb.call(lambda: 42)
            assert result.output == 42

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        async with CircuitBreaker() as cb:

            async def noop() -> None:
                pass

            result = await cb.acall(noop)
            assert result.quality_score == 1.0


class TestTransitionCallback:
    def test_callback_invoked(self) -> None:
        transitions: list[object] = []
        cb = CircuitBreaker(on_transition=transitions.append)
        cb.trip()
        assert len(transitions) == 1
        assert transitions[0].event == TransitionEvent.MANUAL_TRIP  # type: ignore[union-attr]

    def test_callback_exception_ignored(self) -> None:
        def bad_callback(t: object) -> None:
            raise RuntimeError("callback fail")

        cb = CircuitBreaker(on_transition=bad_callback)
        cb.trip()  # Should not raise
        assert cb.state == BreakerState.OPEN


class TestSnapshot:
    def test_snapshot_is_immutable(self) -> None:
        cb = CircuitBreaker()
        snap = cb.snapshot()
        assert snap.state == BreakerState.CLOSED
        with pytest.raises(AttributeError):
            snap.state = BreakerState.OPEN  # type: ignore[misc]

    def test_snapshot_reflects_state(self) -> None:
        cb = CircuitBreaker()
        cb.call(lambda: "ok")
        snap = cb.snapshot()
        assert snap.call_count == 1
        assert snap.failure_count == 0


class TestThreadSafety:
    def test_concurrent_calls(self) -> None:
        # Use quality=1.0 so breaker stays CLOSED
        cb = CircuitBreaker(quality_fn=ContinuousQuality(key=_always_one))
        errors: list[Exception] = []
        results: list[object] = []

        def worker() -> None:
            try:
                for _ in range(20):
                    result = cb.call(lambda: "ok")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 80
        assert cb.snapshot().call_count == 80


class TestTStatistics:
    """Test the pure-math helper functions."""

    def test_regularized_incomplete_beta_bounds(self) -> None:
        assert _regularized_incomplete_beta(1, 1, 0.0) == 0.0
        assert _regularized_incomplete_beta(1, 1, 1.0) == 1.0

    def test_regularized_incomplete_beta_midpoint(self) -> None:
        result = _regularized_incomplete_beta(1, 1, 0.5)
        assert abs(result - 0.5) < 1e-10

    def test_t_cdf_symmetry(self) -> None:
        assert abs(_t_cdf(0.0, 10) - 0.5) < 1e-10

    def test_t_cdf_tails(self) -> None:
        assert _t_cdf(100.0, 10) > 0.999
        assert _t_cdf(-100.0, 10) < 0.001

    def test_one_sample_t_test_obvious_case(self) -> None:
        samples = [0.95, 0.92, 0.98, 0.91, 0.96]
        p = _one_sample_t_test_pvalue(samples, mu_0=0.5)
        assert p < 0.01

    def test_one_sample_t_test_null_case(self) -> None:
        samples = [0.5, 0.5, 0.5, 0.5, 0.5]
        p = _one_sample_t_test_pvalue(samples, mu_0=0.5)
        assert p >= 0.5

    def test_one_sample_t_test_insufficient_samples(self) -> None:
        p = _one_sample_t_test_pvalue([0.9], mu_0=0.5)
        assert p == 1.0
