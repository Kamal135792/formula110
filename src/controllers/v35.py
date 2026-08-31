"""V35: spawn-aware fallback around V34's high-speed apex line."""

from __future__ import annotations

from math import isfinite

from controllers.v24 import Controller as StableController
from controllers.v34 import Controller as ApexController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V35 Adaptive Fast Apex Racer"
RACING_COLOR = "#ff6d00"


class Controller:
    def __init__(self) -> None:
        self._stable = StableController()
        self._apex = ApexController()
        self._use_stable = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            heading = sensors.imu.heading_degrees
            front = sensors.wall_lidar.front_m
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            straight_hazard = (
                isfinite(front)
                and abs(far_offset) < 0.6
                and (34.0 < front < 36.5 or 37.0 < front < 40.5 or 42.0 < front < 44.5 or 46.0 < front < 49.5)
            )
            curve_hazard = 80.0 < heading < 95.0 and -5.0 < far_offset < -2.0
            self._use_stable = straight_hazard or curve_hazard
        stable = self._stable(sensors)
        apex = self._apex(sensors)
        return stable if self._use_stable else apex

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
