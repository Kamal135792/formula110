"""V18: one continuous policy with a dynamically selected speed envelope."""

from __future__ import annotations

from math import isfinite

from controllers.v0 import Controller as V0Controller
from controllers.v0 import PlannerConfig
from controllers.v8 import Controller as StableController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V18 Continuous Boost Racer"
RACING_COLOR = "#6366f1"
SAFE_CONFIG = PlannerConfig(speed_numerator=36.0, curve_speed_gain=1.0)
BOOST_CONFIG = PlannerConfig(speed_numerator=37.0, curve_speed_gain=1.0)
SAFE_LAUNCH_TICKS = 360
STRAIGHT_CURVATURE_LIMIT = 0.03


def _severity(sensors: RobotSensors) -> float:
    offsets = (*sensors.camera.lookahead_offsets_m, 0.0, 0.0, 0.0)
    return max(
        abs(sensors.camera.heading_error_degrees) / 55.0,
        0.62 * abs(offsets[0]) / 4.0,
        0.90 * abs(offsets[1]) / 9.0,
        1.25 * abs(offsets[2]) / 16.0,
    )


class Controller:
    def __init__(self) -> None:
        self._stable = StableController()
        self._racer = V0Controller(planner_config=SAFE_CONFIG)
        self._safe_launch = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            front = sensors.wall_lidar.front_m
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            self._safe_launch = isfinite(front) and 30.0 < front < 40.0 and abs(far_offset) < 0.6

        stable = self._stable(sensors)
        well_aligned = (
            sensors.camera.visible
            and abs(sensors.camera.center_offset_m) < 0.55
            and abs(sensors.camera.heading_error_degrees) < 4.0
            and _severity(sensors) < STRAIGHT_CURVATURE_LIMIT
        )
        self._racer.planner_config = BOOST_CONFIG if well_aligned else SAFE_CONFIG
        command = self._racer(sensors)
        if self._safe_launch and sensors.tick < SAFE_LAUNCH_TICKS:
            return stable
        return command

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
