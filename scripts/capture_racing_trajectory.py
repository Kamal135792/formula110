"""Capture a controller's stable recurring lateral/speed trajectory by track bin."""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean
from typing import cast

from train_v0 import SoloRaceRunner

from controllers.track_localizer import TrackLocalizer
from racing import RobotCommand, RobotSensors, load_student_submission
from racing.student.api import RobotController


class CaptureController:
    def __init__(
        self,
        controller: RobotController,
        *,
        bin_width_m: float,
        warmup_distance_m: float,
    ) -> None:
        self._controller = controller
        self._localizer = TrackLocalizer()
        self._bin_width_m = bin_width_m
        self._warmup_distance_m = warmup_distance_m
        self.centers: dict[int, list[float]] = defaultdict(list)
        self.speeds: dict[int, list[float]] = defaultdict(list)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        if sensors.odometry.distance_m >= self._warmup_distance_m and sensors.camera.visible:
            index = int(progress / self._bin_width_m)
            self.centers[index].append(sensors.camera.center_offset_m)
            self.speeds[index].append(sensors.odometry.speed_mps)
        return self._controller(sensors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--bin-width", type=float, default=5.0)
    parser.add_argument("--warmup-distance", type=float, default=250.0)
    args = parser.parse_args()

    submission = load_student_submission(args.module)
    copy_for_car = getattr(submission.controller, "copy_for_car", None)
    controller = cast(RobotController, copy_for_car()) if callable(copy_for_car) else submission.controller
    capture = CaptureController(
        controller,
        bin_width_m=args.bin_width,
        warmup_distance_m=args.warmup_distance,
    )
    runner = SoloRaceRunner()
    try:
        result = runner.run(capture, seed=args.seed, duration_seconds=args.seconds)
    finally:
        runner.close()

    indices = sorted(set(capture.centers) & set(capture.speeds))
    print("progress_m,center_offset_m,speed_mps,samples")
    for index in indices:
        print(
            f"{(index + 0.5) * args.bin_width:.3f},"
            f"{mean(capture.centers[index]):.6f},"
            f"{mean(capture.speeds[index]):.6f},"
            f"{len(capture.centers[index])}"
        )
    print(
        f"result progress={result.raw_progress_m:.3f} best_lap={result.best_lap_seconds} "
        f"damage={result.damage:.9f} wall={result.wall_contact_seconds:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
