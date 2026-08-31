"""V17: adaptive-launch V12 with a straight-only 40 m/s boost policy."""

from __future__ import annotations

from math import isfinite

from controllers.v0 import Controller as V0Controller
from controllers.v0 import PlannerConfig
from controllers.v8 import Controller as StableController
from controllers.v12 import Controller as SprintController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V17 Straight Boost Racer"
RACING_COLOR = "#d946ef"
SAFE_LAUNCH_TICKS = 360
STRAIGHT_CURVATURE_LIMIT = 0.03


def _curve_severity(sensors: RobotSensors) -> float:
    offsets = sensors.camera.lookahead_offsets_m
    padded = (*offsets, 0.0, 0.0, 0.0)
    return max(
        abs(sensors.camera.heading_error_degrees) / 55.0,
        0.62 * abs(padded[0]) / 4.0,
        0.90 * abs(padded[1]) / 9.0,
        1.25 * abs(padded[2]) / 16.0,
    )


class Controller:
    def __init__(self) -> None:
        self._stable = StableController()
        self._sprint = SprintController()
        self._boost = V0Controller(planner_config=PlannerConfig(speed_numerator=37.0, curve_speed_gain=1.0))
        self._safe_launch = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            front = sensors.wall_lidar.front_m
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            self._safe_launch = isfinite(front) and 30.0 < front < 40.0 and abs(far_offset) < 0.6

        stable = self._stable(sensors)
        sprint = self._sprint(sensors)
        boost = self._boost(sensors)
        if self._safe_launch and sensors.tick < SAFE_LAUNCH_TICKS:
            return stable

        well_aligned = (
            sensors.camera.visible
            and abs(sensors.camera.center_offset_m) < 0.55
            and abs(sensors.camera.heading_error_degrees) < 4.0
            and _curve_severity(sensors) < STRAIGHT_CURVATURE_LIMIT
        )
        throttle = sprint.throttle
        if well_aligned and sprint.throttle > 0.0:
            throttle = max(sprint.throttle, boost.throttle)
        return RobotCommand(throttle=throttle, steer=sprint.steer)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
