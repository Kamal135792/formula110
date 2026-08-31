"""Find the shortest safe stabilization for V53's hazardous start signatures."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS

HAZARD_SEEDS = (10, 13, 21)
STABILIZATION_TICKS = (60, 90, 120, 150, 180, 240, 300)


def main() -> int:
    runner = SoloRaceRunner()
    try:
        for ticks in STABILIZATION_TICKS:
            results: list[SoloEpisodeResult] = []
            for seed in HAZARD_SEEDS:
                controller = Controller(
                    corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                    curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
                    speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
                    straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
                    hazard_stable_ticks=ticks,
                )
                results.append(runner.run(controller, seed=seed, duration_seconds=40.0))
            summary = summarize(results)
            lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
            print(
                f"ticks={ticks:3d} seconds={ticks / 60:4.1f} progress={summary.mean_progress_m:6.1f} "
                f"score={summary.score:7.2f} min={summary.minimum_progress_m:6.1f} "
                f"lap={lap} wall={summary.total_wall_contact_seconds:5.2f} off={summary.total_off_track_seconds:6.2f}",
                flush=True,
            )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
