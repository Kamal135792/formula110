"""V20: track-signature mixture of stable and 36.4 m/s policies."""

from __future__ import annotations

from controllers.v8 import Controller as StableController
from controllers.v12 import Controller as SafeSprintController
from controllers.v16 import Controller as FastSprintController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V20 Section-Aware Racer"
RACING_COLOR = "#22d3ee"
SAFE_LAUNCH_TICKS = 360


class Controller:
    def __init__(self) -> None:
        self._stable = StableController()
        self._safe_sprint = SafeSprintController()
        self._fast_sprint = FastSprintController()
        self._curve_spawn = False
        self._safe_region = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            heading = sensors.imu.heading_degrees
            self._curve_spawn = abs(far_offset) > 3.0
            self._safe_region = 70.0 < heading < 100.0 and -5.0 < far_offset < -2.0

        stable = self._stable(sensors)
        safe_sprint = self._safe_sprint(sensors)
        fast_sprint = self._fast_sprint(sensors)
        if self._safe_region:
            return safe_sprint
        if self._curve_spawn and sensors.tick < SAFE_LAUNCH_TICKS:
            return stable
        return fast_sprint

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
