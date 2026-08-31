"""Fine-search speed scheduling around the successful V43 partition policy."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v43 import LEARNED_CURVE_SPEED_GAINS

SEEDS = (110, 2026, 42, 73, 500, 901)
SECTOR_CANDIDATES = (
    (1.04, 1.08, 1.12, 1.16, 1.20, 1.28),
    (0.62, 0.68, 0.73, 0.78, 0.82, 0.86),
    (0.62, 0.68, 0.73, 0.78, 0.82, 0.86),
    (0.62, 0.68, 0.73, 0.78, 0.82, 0.86),
    (0.62, 0.68, 0.73, 0.78, 0.82, 0.86),
)


def main() -> int:
    gains = list(LEARNED_CURVE_SPEED_GAINS)
    runner = SoloRaceRunner()
    try:
        for sector, candidate_values in enumerate(SECTOR_CANDIDATES):
            candidates: list[tuple[float, float]] = []
            for candidate_gain in candidate_values:
                trial_gains = gains.copy()
                trial_gains[sector] = candidate_gain
                results: list[SoloEpisodeResult] = []
                for seed in SEEDS:
                    controller = Controller(
                        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                        curve_speed_gains=tuple(trial_gains),
                    )
                    results.append(runner.run(controller, seed=seed, duration_seconds=28.0))
                summary = summarize(results)
                safety_penalty = 10.0 * max(0.0, 600.0 - summary.minimum_progress_m)
                objective = -summary.mean_progress_m + safety_penalty + 100.0 * summary.mean_damage
                candidates.append((objective, candidate_gain))
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"sector={sector} gain={candidate_gain:5.3f} progress={summary.mean_progress_m:6.1f} "
                    f"objective={objective:7.1f} score={summary.score:7.2f} "
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
