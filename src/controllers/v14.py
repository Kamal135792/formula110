"""V14: spawn-aware launch selection followed by the V12 sprint policy."""

from __future__ import annotations

from math import isfinite

from controllers.v8 import Controller as StableController
from controllers.v12 import Controller as SprintController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V14 Adaptive Launch Racer"
RACING_COLOR = "#8b5cf6"
SAFE_LAUNCH_TICKS = 360


class Controller:
    def __init__(self) -> None:
        self._stable = StableController()
        self._sprint = SprintController()
        self._use_safe_launch = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            front = sensors.wall_lidar.front_m
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            self._use_safe_launch = isfinite(front) and 30.0 < front < 40.0 and abs(far_offset) < 0.6
        stable_command = self._stable(sensors)
        sprint_command = self._sprint(sensors)
        if self._use_safe_launch and sensors.tick < SAFE_LAUNCH_TICKS:
            return stable_command
        return sprint_command

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
