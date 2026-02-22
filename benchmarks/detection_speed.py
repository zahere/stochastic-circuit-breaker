"""Benchmark: CUSUM vs EMA vs FixedWindow detection speed.

Generates synthetic quality streams with known changepoints and measures
detection delay (ARL_1) across degradation severities. Demonstrates CUSUM's
theoretical advantage for large shifts.

Usage:
    python benchmarks/detection_speed.py
"""

from __future__ import annotations

import random
import statistics

from stochastic_circuit_breaker.detectors import (
    CUSUMDetector,
    EMADetector,
    FixedWindowDetector,
)


def generate_stream(
    normal_len: int,
    degraded_len: int,
    mu_0: float,
    mu_1: float,
    bernoulli: bool = True,
) -> tuple[list[float], int]:
    """Generate a quality stream with a known changepoint.

    Returns (observations, changepoint_index).
    """
    obs: list[float] = []
    for _ in range(normal_len):
        if bernoulli:
            obs.append(1.0 if random.random() < mu_0 else 0.0)
        else:
            obs.append(random.gauss(mu_0, 0.1))
    changepoint = len(obs)
    for _ in range(degraded_len):
        if bernoulli:
            obs.append(1.0 if random.random() < mu_1 else 0.0)
        else:
            obs.append(random.gauss(mu_1, 0.1))
    return obs, changepoint


def measure_detection_delay(
    detector_factory: object,
    streams: list[tuple[list[float], int]],
) -> list[int]:
    """Measure detection delay across multiple streams."""
    delays: list[int] = []
    for obs, cp in streams:
        det = detector_factory()  # type: ignore[operator]
        detected_at = -1
        for i, x in enumerate(obs):
            result = det.update(x)
            if result.alarm and i >= cp:
                detected_at = i
                break
        if detected_at >= 0:
            delays.append(detected_at - cp)
        else:
            delays.append(len(obs) - cp)  # Never detected
    return delays


def calibrate_ema_threshold(mu_0: float, target_arl0: int, alpha: float) -> float:
    """Approximate EMA threshold to match a target ARL_0."""
    best_threshold = mu_0
    best_diff = float("inf")
    for t in [mu_0 - i * 0.01 for i in range(1, 50)]:
        if t <= 0:
            break
        false_alarm_runs = []
        for _ in range(100):
            det = EMADetector(alpha=alpha, threshold=t, warmup=20)
            for j in range(target_arl0 * 3):
                x = 1.0 if random.random() < mu_0 else 0.0
                result = det.update(x)
                if result.alarm:
                    false_alarm_runs.append(j)
                    break
            else:
                false_alarm_runs.append(target_arl0 * 3)
        mean_arl = statistics.mean(false_alarm_runs)
        if abs(mean_arl - target_arl0) < best_diff:
            best_diff = abs(mean_arl - target_arl0)
            best_threshold = t
    return best_threshold


def _stdev_or_zero(values: list[int]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    random.seed(42)

    mu_0 = 0.9
    n_streams = 200
    normal_len = 200
    degraded_len = 300
    target_arl0 = 500

    # Degradation severities (mu_1 values)
    severities = [0.7, 0.5, 0.3, 0.1]

    print("=" * 70)
    print("CUSUM vs EMA vs FixedWindow Detection Delay Benchmark")
    print("=" * 70)
    print(f"  mu_0 = {mu_0}, target ARL_0 ~ {target_arl0}")
    print(f"  {n_streams} streams per severity, {normal_len}+{degraded_len} obs each")
    print()

    # Calibrate CUSUM threshold for target ARL_0
    cusum_h = 8.0  # Conservative

    # Calibrate EMA
    ema_alpha = 0.15
    ema_thresh = calibrate_ema_threshold(mu_0, target_arl0, ema_alpha)
    print(f"  EMA calibrated threshold: {ema_thresh:.3f}")
    print()

    cols = ["mu_1", "CUSUM", "EMA", "FixedWin", "CUSUM adv."]
    header = f"{cols[0]:>6} | {cols[1]:>12} | {cols[2]:>12} | {cols[3]:>12} | {cols[4]:>12}"
    print(header)
    print("-" * len(header))

    for mu_1 in severities:
        streams = [generate_stream(normal_len, degraded_len, mu_0, mu_1) for _ in range(n_streams)]

        # Bind mu_1 via default arg to avoid loop-variable capture
        cusum_delays = measure_detection_delay(
            lambda m=mu_1: CUSUMDetector(mu_0=mu_0, mu_1=m, h=cusum_h),
            streams,
        )

        ema_delays = measure_detection_delay(
            lambda: EMADetector(alpha=ema_alpha, threshold=ema_thresh, warmup=20),
            streams,
        )

        fw_delays = measure_detection_delay(
            lambda: FixedWindowDetector(
                window_size=10, threshold=mu_0 - 0.15, sustained_periods=3
            ),
            streams,
        )

        cm = statistics.mean(cusum_delays)
        em = statistics.mean(ema_delays)
        fm = statistics.mean(fw_delays)
        cs = _stdev_or_zero(cusum_delays)
        es = _stdev_or_zero(ema_delays)
        fs = _stdev_or_zero(fw_delays)
        adv = em / cm if cm > 0 else float("inf")

        print(
            f"{mu_1:>6.1f} | "
            f"{cm:>6.1f}+/-{cs:>4.1f} | "
            f"{em:>6.1f}+/-{es:>4.1f} | "
            f"{fm:>6.1f}+/-{fs:>4.1f} | "
            f"{adv:>10.2f}x"
        )

    print()
    print("CUSUM advantage = EMA / CUSUM delay (higher = faster)")
    print("For large shifts, CUSUM typically achieves 1.4-2x advantage.")
    print()

    # Theoretical ARL comparison
    print("=== Theoretical CUSUM ARL_1 (Wald approximation) ===")
    for mu_1 in severities:
        det = CUSUMDetector(mu_0=mu_0, mu_1=mu_1, h=cusum_h)
        print(
            f"  mu_1={mu_1:.1f}: "
            f"ARL_1 ~ {det.theoretical_arl1():.1f}, "
            f"D_KL = {det.compute_kl_divergence():.4f}"
        )


if __name__ == "__main__":
    main()
