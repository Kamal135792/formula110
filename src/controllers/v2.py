"""V2: curvature-adaptive steering applied to the learned V0 racer."""

from __future__ import annotations

from controllers.v0 import Controller as V0Controller
from racing import RobotCommand, RobotSensors

RACING_NAME = "V2 Apex Racer"
RACING_COLOR = "#a855f7"


class Controller:
    def __init__(self) -> None:
        self._base = V0Controller()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        command = self._base(sensors)
        if sensors.contact.wall > 0.0 or not sensors.camera.visible:
            return command
        curvature = min(1.0, abs(sensors.camera.heading_error_degrees) / 35.0)
        lateral_error = min(1.0, abs(sensors.camera.center_offset_m) / 2.0)
        gain = 1.0 + 0.10 * curvature + 0.04 * lateral_error
        return RobotCommand(throttle=command.throttle, steer=max(-1.0, min(1.0, gain * command.steer)))

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
