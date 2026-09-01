"""Sample sensor-derived curvature/heading-error telemetry to calibrate a turn/straight split.

`sensors.camera.lookahead_offsets_m` already encodes track curvature relative
to the car (that is the whole point of the processed camera readings), so a
turn/straight classifier can be built from sensors alone -- no privileged
track state needed. This script drives a controller through several seeds,
logs the resulting curvature estimate and heading error, and prints
percentiles so a sensible threshold can be picked by inspection.

Usage:
    uv run python scripts/analyze_segment_curvature.py --student-module controllers.model_based_mpc
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from controllers.dynamics.segment import FAR_LOOKAHEAD_DISTANCE_M, estimate_curvature_degrees_per_m
from racing import RobotSensors, load_student_controller, run_headless_head_to_head
from racing.race.head_to_head import HeadToHeadRaceEntry
from racing.race.rules import HeadToHeadRaceRules

DEFAULT_SEEDS: tuple[int, ...] = (42, 110, 271)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample curvature/heading-error telemetry for calibration.")
    parser.add_argument("--student-module", default="controllers.model_based_mpc")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--round-seconds", type=float, default=25.0)
    parser.add_argument("--min-speed-mps", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    curvatures: list[float] = []
    heading_errors: list[float] = []

    for seed in args.seeds:
        controller = load_student_controller(args.student_module)

        def on_sample(_entry: HeadToHeadRaceEntry, sensors: RobotSensors) -> None:
            if sensors.odometry.speed_mps <= args.min_speed_mps:
                return
            curvatures.append(estimate_curvature_degrees_per_m(sensors))
            heading_errors.append(sensors.camera.heading_error_degrees)

        run_headless_head_to_head(
            challenger_controller=controller,
            incumbent_controller=controller,
            race_count=1,
            round_seconds=args.round_seconds,
            random_seed=int(seed),
            rules=HeadToHeadRaceRules(marshal_enabled=False),
            sensor_sample_callback=on_sample,
        )

    curvature_array = np.abs(np.asarray(curvatures))
    heading_error_array = np.abs(np.asarray(heading_errors))
    print(f"samples: {len(curvature_array)} (lookahead distance {FAR_LOOKAHEAD_DISTANCE_M:.0f}m)")
    print(f"{'percentile':>10} {'|curvature| deg/m':>20} {'|heading_error| deg':>20}")
    for percentile in (25, 50, 60, 70, 75, 80, 85, 90, 95, 99):
        curvature_value = float(np.percentile(curvature_array, percentile))
        heading_value = float(np.percentile(heading_error_array, percentile))
        print(f"{percentile:>10} {curvature_value:>20.4f} {heading_value:>20.2f}")
    equivalent_radius_m = [
        (1.0 / math.radians(c)) if c > 1e-6 else float("inf") for c in (0.5, 1.0, 2.0, 3.0, 5.0)
    ]
    print("\nfor reference, curvature -> implied turn radius:")
    for curvature_deg_per_m, radius_m in zip((0.5, 1.0, 2.0, 3.0, 5.0), equivalent_radius_m, strict=True):
        print(f"  {curvature_deg_per_m:.1f} deg/m -> radius {radius_m:.1f} m")


if __name__ == "__main__":
    main()
