"""Print the initial public sensor signature for deterministic spawn seeds."""

from __future__ import annotations

import argparse

from train_v0 import SoloRaceRunner

from controllers.track_localizer import TrackLocalizer
from racing import RobotCommand, RobotSensors


class CaptureController:
    def __init__(self) -> None:
        self.first: RobotSensors | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if self.first is None:
            self.first = sensors
        return RobotCommand()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="110,2026,42,73,500,901,1337,4096,7777,9999")
    args = parser.parse_args()
    runner = SoloRaceRunner()
    try:
        for seed in (int(value) for value in args.seeds.split(",")):
            controller = CaptureController()
            runner.run(controller, seed=seed, duration_seconds=1.0 / 60.0)
            sensors = controller.first
            if sensors is None:
                continue
            offsets = sensors.camera.lookahead_offsets_m
            progress = TrackLocalizer().update(sensors)
            print(
                f"seed={seed} progress={progress:.2f} heading={sensors.imu.heading_degrees:.3f} "
                f"camera_heading={sensors.camera.heading_error_degrees:.3f} "
                f"offsets={tuple(round(value, 3) for value in offsets)} "
                f"walls=({sensors.wall_lidar.left_m:.2f},{sensors.wall_lidar.front_m:.2f},"
                f"{sensors.wall_lidar.right_m:.2f})"
            )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
