"""V19: curve-spawn stabilization followed by V16's 36.4 m/s policy."""

from __future__ import annotations

from controllers.v8 import Controller as StableController
from controllers.v16 import Controller as SprintController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V19 Adaptive 36.4mps Racer"
RACING_COLOR = "#fb7185"
SAFE_LAUNCH_TICKS = 360


class Controller:
    def __init__(self) -> None:
        self._stable = StableController()
        self._sprint = SprintController()
        self._curve_spawn = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            self._curve_spawn = abs(far_offset) > 3.0
        stable = self._stable(sensors)
        sprint = self._sprint(sensors)
        if self._curve_spawn and sensors.tick < SAFE_LAUNCH_TICKS:
            return stable
        return sprint

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
