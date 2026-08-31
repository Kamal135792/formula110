"""Search sector speed envelopes around V70's zero-damage racing line."""

from __future__ import annotations

from statistics import mean

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v69 import Controller
from controllers.v70 import RECOMBINED_INSIDE_OFFSETS_M

SEEDS = (3, 10, 21, 31, 44, 64, 68, 86, 96, 100)
CANDIDATES = (
    ("baseline", ROBUST_LIMIT_CURVE_SPEED_GAINS, 37.5),
    ("s0_fast", (1.10, 0.48, 0.40, 0.58, 0.78), 37.5),
    ("s1_fast", (1.16, 0.44, 0.40, 0.58, 0.78), 37.5),
    ("s2_fast", (1.16, 0.48, 0.34, 0.58, 0.78), 37.5),
    ("s3_fast", (1.16, 0.48, 0.40, 0.53, 0.78), 37.5),
    ("s4_fast", (1.16, 0.48, 0.40, 0.58, 0.72), 37.5),
    ("combined", (1.12, 0.44, 0.36, 0.54, 0.74), 37.5),
    ("straight38", ROBUST_LIMIT_CURVE_SPEED_GAINS, 38.0),
    ("straight39", ROBUST_LIMIT_CURVE_SPEED_GAINS, 39.0),
    ("combo38", (1.12, 0.44, 0.36, 0.54, 0.74), 38.0),
)


def main() -> int:
    runner = SoloRaceRunner()
    rows: list[tuple[int, float, float, str, tuple[float, ...], float]] = []
    try:
        for name, gains, straight_speed in CANDIDATES:
            results: list[SoloEpisodeResult] = [
                runner.run(
                    Controller(
                        recurring_corner_offsets_m=RECOMBINED_INSIDE_OFFSETS_M,
                        recurring_curve_speed_gains=gains,
                        recurring_straight_speed_mps=straight_speed,
                    ),
                    seed=seed,
                    duration_seconds=20.0,
                )
                for seed in SEEDS
            ]
            summary = summarize(results)
            lap_times = [result.best_lap_seconds for result in results if result.best_lap_seconds is not None]
            mean_lap = mean(lap_times) if lap_times else 999.0
            damage_count = sum(result.damage > 0.0 for result in results)
            rows.append((damage_count, mean_lap, -summary.mean_progress_m, name, gains, straight_speed))
            print(
                f"{name:>10} damage_seeds={damage_count} max_damage={max(result.damage for result in results):.6f} "
                f"progress={summary.mean_progress_m:.2f} min={summary.minimum_progress_m:.2f} "
                f"mean_lap={mean_lap:.4f} best={summary.best_lap_seconds}"
            )
    finally:
        runner.close()
    print("ranked (hard safety, then lap time):")
    for damage_count, mean_lap, negative_progress, name, gains, straight_speed in sorted(rows):
        print(
            f"{name:>10} damage_seeds={damage_count} mean_lap={mean_lap:.4f} "
            f"progress={-negative_progress:.2f} speed={straight_speed:.1f} gains={gains}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
