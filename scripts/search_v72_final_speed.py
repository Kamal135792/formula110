"""Retune final-corner and straight speed for V72's wider exit line."""

from __future__ import annotations

from statistics import mean

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v72 import Controller

SEEDS = (3, 10, 31, 64, 96, 100)


def main() -> int:
    runner = SoloRaceRunner()
    rows: list[tuple[int, float, float, float, float]] = []
    try:
        for final_gain in (0.78, 0.74, 0.70, 0.66, 0.62):
            for straight_speed in (37.5, 39.0):
                results: list[SoloEpisodeResult] = [
                    runner.run(
                        Controller(
                            final_curve_speed_gain=final_gain,
                            straight_speed_mps=straight_speed,
                        ),
                        seed=seed,
                        duration_seconds=20.0,
                    )
                    for seed in SEEDS
                ]
                summary = summarize(results)
                laps = [result.best_lap_seconds for result in results if result.best_lap_seconds is not None]
                mean_lap = mean(laps) if laps else 999.0
                damage_count = sum(result.damage > 0.0 for result in results)
                rows.append((damage_count, mean_lap, -summary.mean_progress_m, final_gain, straight_speed))
                print(
                    f"gain={final_gain:.2f} straight={straight_speed:.1f} damage_seeds={damage_count} "
                    f"max_damage={max(result.damage for result in results):.6f} "
                    f"progress={summary.mean_progress_m:.2f} min={summary.minimum_progress_m:.2f} "
                    f"mean_lap={mean_lap:.4f} best={summary.best_lap_seconds}"
                )
    finally:
        runner.close()
    print("ranked (hard safety, then lap time):")
    for damage_count, mean_lap, negative_progress, final_gain, straight_speed in sorted(rows):
        print(
            f"gain={final_gain:.2f} straight={straight_speed:.1f} damage_seeds={damage_count} "
            f"mean_lap={mean_lap:.4f} progress={-negative_progress:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
