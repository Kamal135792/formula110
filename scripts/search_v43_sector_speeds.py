"""Coordinate-search curvature speed independently in each track sector."""

from __future__ import annotations

from statistics import mean

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import SECTOR_BOUNDARIES_M, Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M

SEEDS = (110, 2026, 42, 73, 500, 901)
CANDIDATE_GAINS = (0.78, 0.86, 0.92, 0.972, 1.04, 1.12)


def _sector_times(controller: Controller, sector: int) -> list[float]:
    start = SECTOR_BOUNDARIES_M[sector]
    end = SECTOR_BOUNDARIES_M[sector + 1]
    times: list[float] = []
    previous_tick: int | None = None
    for boundary, tick in controller.sector_crossings:
        if boundary == start:
            previous_tick = tick
        elif boundary == end and previous_tick is not None:
            times.append((tick - previous_tick) / 60.0)
            previous_tick = None
    return times


def main() -> int:
    gains = [0.972] * 5
    runner = SoloRaceRunner()
    try:
        for sector in range(5):
            candidates: list[tuple[float, float]] = []
            for candidate_gain in CANDIDATE_GAINS:
                trial_gains = gains.copy()
                trial_gains[sector] = candidate_gain
                results: list[SoloEpisodeResult] = []
                sector_times: list[float] = []
                for seed in SEEDS:
                    controller = Controller(
                        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                        curve_speed_gains=tuple(trial_gains),
                    )
                    results.append(runner.run(controller, seed=seed, duration_seconds=28.0))
                    sector_times.extend(_sector_times(controller, sector))
                summary = summarize(results)
                segment_time = mean(sector_times) if sector_times else 99.0
                safety_penalty = 10.0 * max(0.0, 600.0 - summary.minimum_progress_m)
                objective = -summary.mean_progress_m + safety_penalty + 100.0 * summary.mean_damage
                candidates.append((objective, candidate_gain))
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"sector={sector} gain={candidate_gain:5.3f} segment={segment_time:6.3f} "
                    f"progress={summary.mean_progress_m:6.1f} objective={objective:7.1f} score={summary.score:7.2f} "
                    f"min={summary.minimum_progress_m:6.1f} lap={lap} wall={summary.total_wall_contact_seconds:5.2f}",
                    flush=True,
                )
            _, gains[sector] = min(candidates)
            print(f"accepted sector={sector}: gains={gains}", flush=True)
    finally:
        runner.close()
    print(f"final gains={tuple(gains)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
