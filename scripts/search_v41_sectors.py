"""Coordinate-search the localized racing line one track sector at a time."""

from __future__ import annotations

from statistics import mean

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import SECTOR_BOUNDARIES_M, Controller

SEEDS = (110, 2026, 42, 73, 500, 901)
CANDIDATE_OFFSETS_M = (0.8, 1.1, 1.4, 1.6, 1.8)


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
    offsets = [1.4] * 5
    runner = SoloRaceRunner()
    try:
        for sector in range(5):
            candidates: list[tuple[float, float, float]] = []
            for candidate_offset in CANDIDATE_OFFSETS_M:
                trial_offsets = offsets.copy()
                trial_offsets[sector] = candidate_offset
                results: list[SoloEpisodeResult] = []
                sector_times: list[float] = []
                for seed in SEEDS:
                    controller = Controller(corner_offsets_m=tuple(trial_offsets))
                    results.append(runner.run(controller, seed=seed, duration_seconds=28.0))
                    sector_times.extend(_sector_times(controller, sector))
                summary = summarize(results)
                segment_time = mean(sector_times) if sector_times else 99.0
                safety_penalty = max(0.0, 620.0 - summary.minimum_progress_m) / 20.0
                objective = segment_time + safety_penalty + 0.06 * summary.mean_damage
                candidates.append((objective, candidate_offset, summary.score))
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"sector={sector} offset={candidate_offset:3.1f} segment={segment_time:6.3f} "
                    f"objective={objective:6.3f} score={summary.score:7.2f} "
                    f"min={summary.minimum_progress_m:6.1f} lap={lap} wall={summary.total_wall_contact_seconds:5.2f}",
                    flush=True,
                )
            _, offsets[sector], _ = min(candidates)
            print(f"accepted sector={sector}: offsets={offsets}", flush=True)
    finally:
        runner.close()
    print(f"final offsets={tuple(offsets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
