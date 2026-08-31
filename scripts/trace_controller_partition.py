"""Trace commands and geometry through a localized track partition."""

from __future__ import annotations

import argparse
from typing import cast

from train_v0 import SoloRaceRunner

from controllers.track_localizer import TrackLocalizer
from racing import RobotCommand, RobotSensors, load_student_submission
from racing.student.api import RobotController


class TraceController:
    def __init__(self, controller: RobotController, *, start_m: float, end_m: float) -> None:
        self._controller = controller
        self._localizer = TrackLocalizer()
        self._start_m = start_m
        self._end_m = end_m
        self._last_bucket = -1

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        command = self._controller(sensors)
        if self._start_m <= progress <= self._end_m:
            bucket = int((progress - self._start_m) * 2.0)
            if bucket != self._last_bucket or sensors.contact.damage > 0.0:
                self._last_bucket = bucket
                print(
                    f"tick={sensors.tick:>3} p={progress:>5.2f} speed={sensors.odometry.speed_mps:>5.2f} "
                    f"right={sensors.wall_lidar.right_m:>5.2f} front={sensors.wall_lidar.front_m:>5.2f} "
                    f"center={sensors.camera.center_offset_m:>5.2f} "
                    f"heading={sensors.camera.heading_error_degrees:>5.1f} "
                    f"cmd={command.throttle:+.2f}/{command.steer:+.2f} damage={sensors.contact.damage:.5f}"
                )
        return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", type=float, default=42.0)
    parser.add_argument("--end", type=float, default=55.0)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    submission = load_student_submission(args.module)
    copy_for_car = getattr(submission.controller, "copy_for_car", None)
    controller = cast(RobotController, copy_for_car()) if callable(copy_for_car) else submission.controller
    runner = SoloRaceRunner()
    try:
        result = runner.run(
            TraceController(controller, start_m=args.start, end_m=args.end),
            seed=args.seed,
            duration_seconds=args.seconds,
        )
    finally:
        runner.close()
    print(f"final damage={result.damage:.6f} progress={result.raw_progress_m:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
