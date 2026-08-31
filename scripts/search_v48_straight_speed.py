"""Search a higher straight-only speed target over V48's learned braking marker."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v48 import OPTIMIZED_SPEED_BOUNDARIES_M

SEEDS = (110, 2026, 42, 73, 500, 901, 7777)
STRAIGHT_TARGETS_MPS = (36.0, 38.0, 40.0, 42.0, 44.0, 48.0)


def main() -> int:
    runner = SoloRaceRunner()
    try:
        for target_speed in STRAIGHT_TARGETS_MPS:
            results: list[SoloEpisodeResult] = []
            for seed in SEEDS:
                controller = Controller(
                    corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                    curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
                    speed_boundaries_m=OPTIMIZED_SPEED_BOUNDARIES_M,
                    straight_speed_numerator=target_speed,
                )
                results.append(runner.run(controller, seed=seed, duration_seconds=32.0))
            summary = summarize(results)
            lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
            print(
                f"target={target_speed:4.1f} progress={summary.mean_progress_m:6.1f} "
                f"score={summary.score:7.2f} min={summary.minimum_progress_m:6.1f} "
                f"lap={lap} speed={summary.mean_max_speed_mps:5.2f} "
                f"wall={summary.total_wall_contact_seconds:5.2f} off={summary.total_off_track_seconds:6.2f}",
                flush=True,
            )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
