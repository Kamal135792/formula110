"""Train the straight schedule for V57's narrow near-line spawn branch."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M
from controllers.v54 import OPTIMIZED_HAZARD_STABILIZATION_TICKS
from controllers.v55 import HAZARD_CURVE_SPEED_GAINS

SENSITIVE_SEEDS = (177, 191, 240, 275, 467, 1337)
STRAIGHT_TARGETS_MPS = (38.0, 40.0, 42.0, 44.0, 46.0)
BRAKING_MARKERS_M = (1.0, 2.0, 3.0, 4.0, 6.0)


def main() -> int:
    runner = SoloRaceRunner()
    try:
        for target_speed in STRAIGHT_TARGETS_MPS:
            for braking_marker in BRAKING_MARKERS_M:
                boundaries = (braking_marker, *AGGRESSIVE_BRAKING_BOUNDARIES_M[1:])
                results: list[SoloEpisodeResult] = []
                for seed in SENSITIVE_SEEDS:
                    controller = Controller(
                        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                        curve_speed_gains=HAZARD_CURVE_SPEED_GAINS,
                        speed_boundaries_m=boundaries,
                        straight_speed_numerator=target_speed,
                        hazard_stable_ticks=OPTIMIZED_HAZARD_STABILIZATION_TICKS,
                    )
                    results.append(runner.run(controller, seed=seed, duration_seconds=40.0))
                summary = summarize(results)
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"target={target_speed:4.1f} brake={braking_marker:3.0f} progress={summary.mean_progress_m:6.1f} "
                    f"score={summary.score:7.2f} min={summary.minimum_progress_m:6.1f} lap={lap} "
                    f"speed={summary.mean_max_speed_mps:5.2f} wall={summary.total_wall_contact_seconds:5.2f} "
                    f"off={summary.total_off_track_seconds:5.2f}",
                    flush=True,
                )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
