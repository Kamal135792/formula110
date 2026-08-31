"""Re-optimize sector racing lines for V46's faster speed schedule."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS

SEEDS = (110, 2026, 42, 73, 500, 901, 7777)
SECTOR_CANDIDATES = (
    (0.40, 0.60, 0.80, 1.00, 1.20),
    (1.30, 1.50, 1.70, 1.80, 1.90, 2.10),
    (0.40, 0.60, 0.80, 1.00, 1.20),
    (1.00, 1.20, 1.40, 1.60, 1.80),
    (1.40, 1.60, 1.80, 2.00, 2.20),
)


def main() -> int:
    offsets = list(LEARNED_CORNER_OFFSETS_M)
    runner = SoloRaceRunner()
    try:
        for sector, candidate_values in enumerate(SECTOR_CANDIDATES):
            candidates: list[tuple[float, float]] = []
            for candidate_offset in candidate_values:
                trial_offsets = offsets.copy()
                trial_offsets[sector] = candidate_offset
                results: list[SoloEpisodeResult] = []
                for seed in SEEDS:
                    controller = Controller(
                        corner_offsets_m=tuple(trial_offsets),
                        curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
                    )
                    results.append(runner.run(controller, seed=seed, duration_seconds=28.0))
                summary = summarize(results)
                safety_penalty = 10.0 * max(0.0, 600.0 - summary.minimum_progress_m)
                objective = -summary.mean_progress_m + safety_penalty + 100.0 * summary.mean_damage
                candidates.append((objective, candidate_offset))
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"sector={sector} offset={candidate_offset:4.2f} progress={summary.mean_progress_m:6.1f} "
                    f"objective={objective:7.1f} score={summary.score:7.2f} "
                    f"min={summary.minimum_progress_m:6.1f} lap={lap} wall={summary.total_wall_contact_seconds:5.2f}",
                    flush=True,
                )
            _, offsets[sector] = min(candidates)
            print(f"accepted sector={sector}: offsets={offsets}", flush=True)
    finally:
        runner.close()
    print(f"final offsets={tuple(offsets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
