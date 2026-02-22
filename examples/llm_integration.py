"""Wrapping LLM API calls with a stochastic circuit breaker.

Shows how to use quality functions that evaluate LLM response quality,
not just success/failure.
"""

from __future__ import annotations

import json
import random
from typing import Any

from stochastic_circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    ContinuousQuality,
)

# --- Simulated LLM responses (replace with real API calls) ---


def simulated_llm_call(prompt: str, *, degraded: bool = False) -> dict[str, Any]:
    """Simulate an LLM API response.

    In production, replace with:
        response = openai.chat.completions.create(...)
        return response.model_dump()
    """
    if degraded and random.random() < 0.6:
        # Simulate degraded responses: malformed JSON, refusals, gibberish
        return {
            "content": random.choice(
                [
                    "I cannot help with that.",
                    '{"partial": true',  # Malformed JSON
                    "Error: rate limited",
                    "",
                ]
            ),
            "usage": {"total_tokens": random.randint(1, 10)},
            "finish_reason": random.choice(["stop", "length"]),
        }
    return {
        "content": json.dumps({"answer": "The result is 42", "confidence": 0.95}),
        "usage": {"total_tokens": random.randint(50, 200)},
        "finish_reason": "stop",
    }


# --- Quality functions for LLM responses ---


def json_compliance_quality(response: dict[str, Any]) -> float:
    """Score based on whether the response contains valid JSON."""
    content = response.get("content", "")
    if not content:
        return 0.0
    try:
        json.loads(content)
        return 1.0
    except json.JSONDecodeError:
        return 0.0


def format_compliance_quality(response: dict[str, Any]) -> float:
    """Combined quality score: JSON validity + non-empty + reasonable length."""
    content = response.get("content", "")
    if not content:
        return 0.0

    score = 0.0

    # JSON parseable?
    try:
        parsed = json.loads(content)
        score += 0.4
        # Has expected fields?
        if isinstance(parsed, dict) and "answer" in parsed:
            score += 0.3
        # Has confidence?
        if isinstance(parsed, dict) and "confidence" in parsed:
            score += 0.3
    except json.JSONDecodeError:
        # Not JSON, check if it's at least a substantive response
        if len(content) > 20 and not content.startswith("Error"):
            score += 0.2

    return min(1.0, score)


# --- Main example ---


def main() -> None:
    cb = CircuitBreaker(
        quality_fn=ContinuousQuality(key=format_compliance_quality),
        config=BreakerConfig(
            mu_0=0.85,  # Expect ~85% quality normally
            mu_1=0.35,  # Degraded quality threshold
            h_warn=4.0,
            h_crit=8.0,
            open_timeout=10.0,
            probe_window=5,
        ),
    )

    # Phase 1: Normal operation
    print("=== Phase 1: Normal LLM operation ===")
    for i in range(15):
        try:
            result = cb.call(simulated_llm_call, "What is 6*7?", degraded=False)
            print(
                f"  [{i:2d}] quality={result.quality_score:.1f} "
                f"state={result.state.value} W_t={result.statistic:.2f}"
            )
        except CircuitOpenError as e:
            print(f"  [{i:2d}] BLOCKED: {e}")

    # Phase 2: Degraded LLM
    print("\n=== Phase 2: LLM provider degraded ===")
    for i in range(15, 40):
        try:
            result = cb.call(simulated_llm_call, "What is 6*7?", degraded=True)
            print(
                f"  [{i:2d}] quality={result.quality_score:.1f} "
                f"state={result.state.value} W_t={result.statistic:.2f}"
            )
        except CircuitOpenError as e:
            print(f"  [{i:2d}] BLOCKED: {e}")

    print(f"\nFinal state: {cb.state.value}")
    print(f"CUSUM statistic: {cb.statistic:.2f}")
    snap = cb.snapshot()
    print(f"Calls: {snap.call_count}, Failures: {snap.failure_count}")


if __name__ == "__main__":
    main()
