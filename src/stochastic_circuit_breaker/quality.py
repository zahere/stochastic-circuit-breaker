"""Quality functions for scoring LLM/agent outputs.

Quality functions map raw outputs to [0, 1] scores that the circuit breaker
uses as observations for change detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class QualityFunction(Protocol):
    """Protocol for quality scoring functions.

    A quality function evaluates an output and returns a score in [0, 1],
    where 1.0 indicates perfect quality and 0.0 indicates complete failure.
    """

    def __call__(self, output: Any) -> float: ...


class BinaryQuality:
    """Binary quality: 1.0 if output is not an exception, 0.0 otherwise.

    This is the default quality function. It treats any non-exception output
    as a success (1.0) and any exception as a failure (0.0).

    The circuit breaker handles exceptions before the quality function is called,
    so in practice this always returns 1.0 for outputs that reach it. Exceptions
    are recorded as 0.0 by the breaker itself.
    """

    def __call__(self, output: Any) -> float:
        return 0.0 if isinstance(output, BaseException) else 1.0


class ThresholdQuality:
    """Quality based on a numeric threshold.

    Returns 1.0 if the output (or extracted value) meets or exceeds the
    threshold, 0.0 otherwise. Useful for confidence scores, token counts, etc.

    Args:
        threshold: Minimum value for quality=1.0.
        key: Optional callable to extract the numeric value from the output.
             If None, the output itself is used.
    """

    def __init__(self, threshold: float, key: Callable[[Any], float] | None = None) -> None:
        self._threshold = threshold
        self._key = key

    def __call__(self, output: Any) -> float:
        try:
            value = self._key(output) if self._key is not None else float(output)
        except (TypeError, ValueError, KeyError, AttributeError):
            return 0.0
        return 1.0 if value >= self._threshold else 0.0


class ContinuousQuality:
    """Extract a continuous [0, 1] quality score from the output.

    Unlike ThresholdQuality which returns binary 0/1, this returns the actual
    score value clamped to [0, 1]. Useful when you have a direct quality metric
    (e.g., BLEU score, cosine similarity, format compliance ratio).

    Args:
        key: Callable to extract the numeric quality score from the output.
    """

    def __init__(self, key: Callable[[Any], float]) -> None:
        self._key = key

    def __call__(self, output: Any) -> float:
        try:
            value = self._key(output)
        except (TypeError, ValueError, KeyError, AttributeError):
            return 0.0
        return max(0.0, min(1.0, value))
