"""Jointly test wider exit arcs with V75's early full-throttle point."""

from __future__ import annotations

from statistics import mean

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v72 import Controller

SEEDS = (3, 10, 31, 64, 96, 100)
CANDIDATES = (
    (-0.09, 156.0),
    (-0.09, 160.0),
    (-0.12, 156.0),
    (-0.12, 160.0),
    (-0.15, 156.0),
    (-0.15, 164.0),
)


def main() -> int:
    runner = SoloRaceRunner()
    rows: list[tuple[int, float, float, float, float]] = []
    try:
        for exit_steer, exit_peak_m in CANDIDATES:
            results: list[SoloEpisodeResult] = [
                runner.run(
                    Controller(
                        exit_steer=exit_steer,
                        exit_peak_m=exit_peak_m,
                        exit_end_m=exit_peak_m + 10.0,
                        straight_speed_mps=39.0,
                        exit_throttle_floor=1.0,
                        exit_throttle_start_m=157.0,
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
            rows.append((damage_count, mean_lap, -summary.mean_progress_m, exit_steer, exit_peak_m))
            print(
                f"exit={exit_steer:+.2f} peak={exit_peak_m:.0f} damage_seeds={damage_count} "
                f"max_damage={max(result.damage for result in results):.6f} "
                f"progress={summary.mean_progress_m:.2f} min={summary.minimum_progress_m:.2f} "
                f"mean_lap={mean_lap:.4f} best={summary.best_lap_seconds}"
            )
    finally:
        runner.close()
    print("ranked (hard safety, then lap time):")
    for damage_count, mean_lap, negative_progress, exit_steer, exit_peak_m in sorted(rows):
        print(
            f"exit={exit_steer:+.2f} peak={exit_peak_m:.0f} damage_seeds={damage_count} "
            f"mean_lap={mean_lap:.4f} progress={-negative_progress:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
