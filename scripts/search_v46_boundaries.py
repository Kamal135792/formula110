"""Coordinate-search the speed-policy handoff positions around V46."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import SECTOR_BOUNDARIES_M, Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS

SEEDS = (110, 2026, 42, 73, 500, 901, 7777)
BOUNDARY_CANDIDATES = (
    (12.0, 15.0, 17.0, 19.0, 22.0),
    (43.0, 46.0, 48.0, 50.0, 53.0),
    (75.0, 78.0, 80.0, 82.0, 85.0),
    (105.0, 108.0, 110.0, 112.0, 115.0),
    (135.0, 138.0, 140.0, 142.0, 145.0),
    (160.0, 163.0, 165.0, 167.0, 170.0),
)


def main() -> int:
    boundaries = list(SECTOR_BOUNDARIES_M)
    runner = SoloRaceRunner()
    try:
        for boundary_index, candidate_values in enumerate(BOUNDARY_CANDIDATES):
            candidates: list[tuple[float, float]] = []
            for candidate_boundary in candidate_values:
                trial_boundaries = boundaries.copy()
                trial_boundaries[boundary_index] = candidate_boundary
                results: list[SoloEpisodeResult] = []
                for seed in SEEDS:
                    controller = Controller(
                        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                        curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
                        speed_boundaries_m=tuple(trial_boundaries),
                    )
                    results.append(runner.run(controller, seed=seed, duration_seconds=28.0))
                summary = summarize(results)
                safety_penalty = 10.0 * max(0.0, 600.0 - summary.minimum_progress_m)
                objective = -summary.mean_progress_m + safety_penalty + 100.0 * summary.mean_damage
                candidates.append((objective, candidate_boundary))
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"boundary={boundary_index} at={candidate_boundary:5.1f} progress={summary.mean_progress_m:6.1f} "
                    f"objective={objective:7.1f} score={summary.score:7.2f} "
                    f"min={summary.minimum_progress_m:6.1f} lap={lap} wall={summary.total_wall_contact_seconds:5.2f}",
                    flush=True,
                )
            _, boundaries[boundary_index] = min(candidates)
            print(f"accepted boundary={boundary_index}: boundaries={boundaries}", flush=True)
    finally:
        runner.close()
    print(f"final boundaries={tuple(boundaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
