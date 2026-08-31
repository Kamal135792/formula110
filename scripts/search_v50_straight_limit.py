"""Fine-search the upper straight-speed limit with earlier braking handoffs."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v48 import OPTIMIZED_SPEED_BOUNDARIES_M

SEEDS = (110, 2026, 42, 73, 500, 901, 7777)
STRAIGHT_TARGETS_MPS = (37.50, 37.75, 38.00, 38.25)
BRAKING_MARKERS_M = (2.0, 4.0, 6.0, 8.0)


def main() -> int:
    runner = SoloRaceRunner()
    try:
        for target_speed in STRAIGHT_TARGETS_MPS:
            for braking_marker in BRAKING_MARKERS_M:
                boundaries = (braking_marker, *OPTIMIZED_SPEED_BOUNDARIES_M[1:])
                results: list[SoloEpisodeResult] = []
                for seed in SEEDS:
                    controller = Controller(
                        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                        curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
                        speed_boundaries_m=boundaries,
                        straight_speed_numerator=target_speed,
                    )
                    results.append(runner.run(controller, seed=seed, duration_seconds=32.0))
                summary = summarize(results)
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"target={target_speed:5.2f} brake={braking_marker:4.1f} "
                    f"progress={summary.mean_progress_m:6.1f} score={summary.score:7.2f} "
                    f"min={summary.minimum_progress_m:6.1f} lap={lap} speed={summary.mean_max_speed_mps:5.2f} "
                    f"wall={summary.total_wall_contact_seconds:5.2f} off={summary.total_off_track_seconds:6.2f}",
                    flush=True,
                )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
