"""Coordinate-search a speed schedule specifically for V54's hazard trajectories."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS
from controllers.v54 import OPTIMIZED_HAZARD_STABILIZATION_TICKS

HAZARD_SEEDS = (10, 13, 21)
SECTOR_CANDIDATES = (
    (1.00, 1.08, 1.16, 1.24, 1.32),
    (0.36, 0.42, 0.48, 0.54, 0.60),
    (0.32, 0.36, 0.40, 0.44, 0.48),
    (0.48, 0.54, 0.58, 0.62, 0.68),
    (0.68, 0.74, 0.78, 0.82, 0.88),
)


def main() -> int:
    gains = list(ROBUST_LIMIT_CURVE_SPEED_GAINS)
    runner = SoloRaceRunner()
    try:
        for sector, candidate_values in enumerate(SECTOR_CANDIDATES):
            candidates: list[tuple[float, float]] = []
            for candidate_gain in candidate_values:
                trial_gains = gains.copy()
                trial_gains[sector] = candidate_gain
                results: list[SoloEpisodeResult] = []
                for seed in HAZARD_SEEDS:
                    controller = Controller(
                        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                        curve_speed_gains=tuple(trial_gains),
                        speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
                        straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
                        hazard_stable_ticks=OPTIMIZED_HAZARD_STABILIZATION_TICKS,
                    )
                    results.append(runner.run(controller, seed=seed, duration_seconds=40.0))
                summary = summarize(results)
                safety_penalty = 10.0 * max(0.0, 950.0 - summary.minimum_progress_m)
                objective = -summary.mean_progress_m + safety_penalty + 100.0 * summary.mean_damage
                candidates.append((objective, candidate_gain))
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"sector={sector} gain={candidate_gain:4.2f} progress={summary.mean_progress_m:6.1f} "
                    f"objective={objective:7.1f} score={summary.score:7.2f} min={summary.minimum_progress_m:6.1f} "
                    f"lap={lap} wall={summary.total_wall_contact_seconds:5.2f}",
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
