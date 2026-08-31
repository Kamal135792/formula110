"""Evaluate diverse recurring-lap path combinations behind V69's safe launch."""

from __future__ import annotations

from statistics import mean

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v69 import Controller

SEEDS = (3, 10, 21, 31, 44, 64, 68, 86, 96, 100)
PATHS = (
    ("incumbent", (0.8, 1.8, 0.8, 1.4, 1.8)),
    ("smooth", (1.2, 1.2, 1.2, 1.2, 1.2)),
    ("wide", (0.4, 1.4, 0.4, 1.0, 1.4)),
    ("late_apex", (0.2, 2.1, 0.4, 1.8, 2.2)),
    ("inside", (1.2, 2.2, 1.2, 1.8, 2.2)),
    ("cross_a", (1.0, 1.8, 1.0, 1.4, 2.0)),
    ("cross_b", (0.6, 2.0, 0.6, 1.6, 2.0)),
    ("cross_c", (1.0, 1.6, 0.6, 1.6, 2.2)),
    ("cross_d", (0.4, 1.8, 1.2, 1.2, 1.6)),
)


def main() -> int:
    runner = SoloRaceRunner()
    ranked: list[tuple[int, float, float, str, tuple[float, ...]]] = []
    try:
        for name, path in PATHS:
            results: list[SoloEpisodeResult] = [
                runner.run(
                    Controller(recurring_corner_offsets_m=path),
                    seed=seed,
                    duration_seconds=20.0,
                )
                for seed in SEEDS
            ]
            summary = summarize(results)
            lap_times = [result.best_lap_seconds for result in results if result.best_lap_seconds is not None]
            damage_count = sum(result.damage > 0.0 for result in results)
            mean_lap = mean(lap_times) if lap_times else 999.0
            ranked.append((damage_count, -summary.mean_progress_m, mean_lap, name, path))
            print(
                f"{name:>10} damage_seeds={damage_count} max_damage={max(result.damage for result in results):.6f} "
                f"progress={summary.mean_progress_m:.2f} min={summary.minimum_progress_m:.2f} "
                f"mean_lap={mean_lap:.4f} best={summary.best_lap_seconds}"
            )
    finally:
        runner.close()
    print("ranked (hard safety first):")
    for damage_count, negative_progress, mean_lap, name, path in sorted(ranked):
        print(
            f"{name:>10} damage_seeds={damage_count} progress={-negative_progress:.2f} "
            f"mean_lap={mean_lap:.4f} {path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
