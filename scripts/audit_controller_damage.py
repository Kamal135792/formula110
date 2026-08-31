"""Report the first controller tick at which each solo run accumulates damage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import cast

from benchmark_controller import _parse_seeds
from train_v0 import SoloRaceRunner

from controllers.track_localizer import TrackLocalizer
from racing import RobotCommand, RobotSensors, load_student_submission
from racing.student.api import RobotController


@dataclass(slots=True)
class DamageAuditController:
    controller: RobotController
    first_damage_tick: int | None = None
    first_wall_tick: int | None = None
    damage_at_first_tick: float = 0.0
    first_damage_progress_m: float | None = None
    first_damage_speed_mps: float | None = None
    first_damage_geometry: tuple[float, float, float, float, float] | None = None
    command_at_first_damage: RobotCommand | None = None
    localizer: TrackLocalizer | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if self.localizer is None:
            self.localizer = TrackLocalizer()
        progress = self.localizer.update(sensors)
        command = self.controller(sensors)
        if self.first_damage_tick is None and sensors.contact.damage > 0.0:
            self.first_damage_tick = sensors.tick
            self.damage_at_first_tick = sensors.contact.damage
            self.first_damage_progress_m = progress
            self.first_damage_speed_mps = sensors.odometry.speed_mps
            self.first_damage_geometry = (
                sensors.wall_lidar.left_m,
                sensors.wall_lidar.front_m,
                sensors.wall_lidar.right_m,
                sensors.camera.center_offset_m,
                sensors.camera.heading_error_degrees,
            )
            self.command_at_first_damage = command
        if self.first_wall_tick is None and sensors.contact.wall > 0.0:
            self.first_wall_tick = sensors.tick
        return command


def _fresh(controller: RobotController) -> RobotController:
    copy_for_car = getattr(controller, "copy_for_car", None)
    return cast(RobotController, copy_for_car()) if callable(copy_for_car) else controller


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("--seeds", type=_parse_seeds, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    submission = load_student_submission(args.module)
    runner = SoloRaceRunner()
    try:
        for seed in args.seeds:
            audit = DamageAuditController(_fresh(submission.controller))
            result = runner.run(audit, seed=seed, duration_seconds=args.seconds)
            first_time = (
                "--" if audit.first_damage_tick is None else f"{audit.first_damage_tick / 60.0:.3f}s"
            )
            wall_time = "--" if audit.first_wall_tick is None else f"{audit.first_wall_tick / 60.0:.3f}s"
            location = (
                "--"
                if audit.first_damage_progress_m is None
                else f"{audit.first_damage_progress_m:.2f}m@{audit.first_damage_speed_mps:.2f}m/s"
            )
            geometry = (
                "--"
                if audit.first_damage_geometry is None
                else "L/F/R={:.2f}/{:.2f}/{:.2f} center={:.2f} heading={:.1f}".format(
                    *audit.first_damage_geometry
                )
            )
            command = (
                "--"
                if audit.command_at_first_damage is None
                else f"cmd={audit.command_at_first_damage.throttle:+.2f}/{audit.command_at_first_damage.steer:+.2f}"
            )
            print(
                f"seed={seed:>5} first_damage={first_time:>8} first_wall={wall_time:>8} "
                f"location={location:>16} "
                f"initial_delta={audit.damage_at_first_tick:.6f} final={result.damage:.6f} "
                f"best_lap={result.best_lap_seconds} {geometry} {command}"
            )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
