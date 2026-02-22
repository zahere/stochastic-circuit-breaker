# Stochastic Circuit Breaker

[![CI](https://github.com/zahere/stochastic-circuit-breaker/actions/workflows/ci.yml/badge.svg)](https://github.com/zahere/stochastic-circuit-breaker/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**A statistically optimal circuit breaker for stochastic systems.**

Every existing circuit breaker library is deterministic: count N failures, trip. This works for databases and HTTP services where failures are binary. It fails for LLM agents and AI pipelines where **retrying is resampling, not replaying** — the same prompt can produce wildly different quality outputs on consecutive calls.

This library implements a 4-state circuit breaker using CUSUM (Cumulative Sum) sequential change detection, which is provably optimal for detecting quality degradation in stochastic systems (Moustakides, 1986).

## Why Deterministic Circuit Breakers Fail for LLMs

Consider an LLM agent that formats responses as JSON:

| Call | Response | Deterministic CB | Stochastic CB |
|------|----------|-------------------|---------------|
| 1 | Valid JSON, high quality | Success | quality = 0.95 |
| 2 | Valid JSON, wrong schema | Success (no exception) | quality = 0.3 |
| 3 | Valid JSON, high quality | Success | quality = 0.92 |
| 4 | Refusal ("I can't help") | Success (no exception) | quality = 0.0 |
| 5 | Valid JSON, truncated | Success (no exception) | quality = 0.4 |

The deterministic CB sees 5 successes. The stochastic CB sees a **quality degradation** — the CUSUM statistic accumulates evidence of a distributional shift and transitions to DEGRADED after call 5.

**Key differences:**

| Feature | Deterministic CB | Stochastic CB |
|---------|------------------|---------------|
| Input signal | Binary (success/fail) | Continuous quality [0,1] |
| Detection | Count N failures | Sequential likelihood ratio |
| Optimality | None | Minimax optimal (Moustakides 1986) |
| False alarm control | Ad hoc threshold | Theoretical ARL_0 guarantee |
| Partial degradation | Invisible | Detected via DEGRADED state |
| Recovery validation | N successes | One-sample t-test |

## Quick Start

```python
from stochastic_circuit_breaker import CircuitBreaker, ContinuousQuality, BreakerConfig

# Define how to score response quality
quality_fn = ContinuousQuality(key=lambda resp: resp.get("confidence", 0.0))

cb = CircuitBreaker(
    quality_fn=quality_fn,
    config=BreakerConfig(mu_0=0.85, mu_1=0.4),
)

# Wrap your LLM calls
result = cb.call(my_llm_function, prompt="Summarize this document")
print(result.state, result.quality_score)
```

## The 4-State Automaton

```
                    W_t >= h_warn               W_t >= h_crit
    ┌────────┐ ──────────────── ┌───────────┐ ──────────────── ┌──────┐
    │ CLOSED │                  │ DEGRADED  │                  │ OPEN │
    └────────┘ ◄─────────────── └───────────┘                  └──────┘
                   W_t == 0                                       │
                                                                  │ timeout
        ┌──────────────────────────────────────────────┐          │
        │ CLOSED │ ◄── t-test pass ── │ PROBING │ ◄────┘
        └────────┘     t-test fail ──► │         │ ──► OPEN
                                       └─────────┘
```

| State | Meaning | Behavior |
|-------|---------|----------|
| **CLOSED** | Normal operation | All calls pass through, quality monitored |
| **DEGRADED** | Early warning | Calls still pass, CUSUM accumulating evidence |
| **OPEN** | Confirmed degradation | Calls blocked with `CircuitOpenError` |
| **PROBING** | Recovery testing | Limited calls allowed, t-test validates recovery |

## CUSUM Theory

The CUSUM detector accumulates the log-likelihood ratio:

```
W_t = max(0, W_{t-1} + log(f_1(X_t) / f_0(X_t)))
```

where `f_0` is the quality distribution under normal operation (H0) and `f_1` is the distribution under degradation (H1). An alarm fires when `W_t >= h`.

**For Bernoulli observations** (success/failure with rates μ₀, μ₁):
- Success: `LLR = log(μ₁/μ₀)` (negative — evidence for H0)
- Failure: `LLR = log((1-μ₁)/(1-μ₀))` (positive — evidence for H1)

**Optimality (Moustakides 1986):** Among all detectors with the same false alarm rate (ARL₀), CUSUM minimizes the worst-case detection delay. This is not an approximation — it is a proven minimax result.

## Configuration

```python
from stochastic_circuit_breaker import BreakerConfig

config = BreakerConfig(
    mu_0=0.9,            # Expected quality under H0 (normal)
    mu_1=0.5,            # Expected quality under H1 (degraded)
    h_warn=3.0,          # CUSUM threshold: CLOSED -> DEGRADED
    h_crit=8.0,          # CUSUM threshold: DEGRADED -> OPEN
    probe_window=10,     # Observations to collect in PROBING
    recovery_alpha=0.05, # t-test significance level for recovery
    open_timeout=30.0,   # Seconds in OPEN before PROBING
    bernoulli=True,      # True for binary, False for Gaussian model
    sigma=0.1,           # Std dev for Gaussian model
)
```

**Choosing thresholds:** `h_warn` and `h_crit` control the tradeoff between detection speed (ARL₁) and false alarm rate (ARL₀). Higher thresholds = fewer false alarms but slower detection. Use `CUSUMDetector.theoretical_arl1()` and `theoretical_arl0()` to estimate.

## Detectors

| Detector | Algorithm | Optimal? | Use Case |
|----------|-----------|----------|----------|
| `CUSUMDetector` | Cumulative sum of LLR | Yes (Moustakides 1986) | Production monitoring |
| `TwoThresholdCUSUM` | CUSUM with warn + crit | Yes | 4-state circuit breaker (default) |
| `EMADetector` | Exponential moving average | No | Simple baseline, fast prototyping |
| `FixedWindowDetector` | Epoch-based window mean | No | Traditional CB comparison |

All detectors implement the `ChangeDetector` protocol:

```python
from stochastic_circuit_breaker import ChangeDetector

def monitor(detector: ChangeDetector, observations: list[float]) -> None:
    for x in observations:
        result = detector.update(x)
        if result.alarm:
            print(f"Degradation detected at W_t = {result.statistic:.2f}")
            break
```

## Quality Functions

Quality functions map raw outputs to [0, 1] scores:

```python
from stochastic_circuit_breaker import BinaryQuality, ThresholdQuality, ContinuousQuality

# Binary: 1.0 if no exception, 0.0 otherwise (default)
binary = BinaryQuality()

# Threshold: 1.0 if value >= threshold, 0.0 otherwise
confidence = ThresholdQuality(threshold=0.7, key=lambda r: r["confidence"])

# Continuous: extract and clamp a [0, 1] score
bleu = ContinuousQuality(key=lambda r: r["bleu_score"])
```

Custom quality functions just need to be callable with signature `(output) -> float`:

```python
def json_quality(response: dict) -> float:
    """Score based on JSON validity and schema compliance."""
    import json
    try:
        data = json.loads(response["content"])
        return 1.0 if "answer" in data else 0.5
    except (json.JSONDecodeError, KeyError):
        return 0.0

cb = CircuitBreaker(quality_fn=json_quality)
```

## Multi-Agent Registry

Monitor a fleet of agents with automatic cleanup:

```python
from stochastic_circuit_breaker import BreakerRegistry, CircuitBreaker

registry = BreakerRegistry()

# Register breakers (uses weak references — auto-cleanup on GC)
for agent_name in ["agent-1", "agent-2", "agent-3"]:
    cb = CircuitBreaker()
    registry.register(agent_name, cb)

# Fleet inspection
print(registry.summary())        # {"closed": 2, "degraded": 1, ...}
print(registry.open_breakers())   # ["agent-3"]

# Detailed snapshots
for name, snap in registry.snapshots().items():
    print(f"{name}: state={snap.state.value}, W_t={snap.statistic:.2f}")
```

## Async Support

```python
import asyncio
from stochastic_circuit_breaker import CircuitBreaker

cb = CircuitBreaker()

async def main():
    result = await cb.acall(my_async_llm_call, prompt="Hello")
    print(result.state, result.quality_score)

asyncio.run(main())
```

## Thread Safety

The circuit breaker is thread-safe. A lock protects state mutations, but the wrapped function executes outside the lock to avoid blocking concurrent callers.

```python
import threading
from stochastic_circuit_breaker import CircuitBreaker

cb = CircuitBreaker()

def worker():
    for _ in range(100):
        result = cb.call(my_function)

threads = [threading.Thread(target=worker) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Benchmarks

Run the detection speed benchmark:

```bash
python benchmarks/detection_speed.py
```

This compares CUSUM vs EMA vs FixedWindow detection delay across degradation severities, matching ARL₀ ≈ 500. CUSUM achieves 1.4–1.6× faster detection for large shifts (D_KL ≥ 0.25), validated across 24,000 synthetic trials.

## Installation

```bash
uv pip install git+https://github.com/zahere/stochastic-circuit-breaker.git
```

**Zero dependencies.** Core uses only Python stdlib (`math`, `threading`, `time`, `weakref`).

## Origin

This library implements the statistical detection layer from
"When Does Topology Matter? Fault-Dependent Resilience in Multi-Agent
LLM Systems" (in preparation) — developed during research that grew
out of production work on [AgentiCraft](https://agenticraft.ai),
an enterprise multi-agent platform.

The core problem: deterministic circuit breakers (count N failures,
trip) are fundamentally wrong for stochastic systems. An LLM that
returns low-quality output 3 times in a row might just be unlucky.
This library gives you statistically optimal detection that
distinguishes genuine degradation from normal variance.

A companion library implementing the reliability polynomial framework
from the same research is available at
[reliability-polynomials](https://github.com/zahere/reliability-polynomials).

## Author

**Zaher Khateeb** — AI/ML Engineer, Founder of [AgentiCraft](https://agenticraft.ai)

Research focus: fault-dependent resilience in multi-agent LLM systems,
stochastic service mesh architecture, formal verification for
distributed agent coordination.

[linkedin.com/in/zahere](https://www.linkedin.com/in/zahere/) ·
[agenticraft.ai](https://agenticraft.ai)

## References

- **Page, E. S.** (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100-115.
- **Moustakides, G. V.** (1986). "Optimal Stopping Times for Detecting Changes in Distributions." *The Annals of Statistics*, 14(4), 1379-1387.
- **Lorden, G.** (1971). "Procedures for Reacting to a Change in Distribution." *The Annals of Mathematical Statistics*, 42(6), 1897-1908.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
