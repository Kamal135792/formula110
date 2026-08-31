"""V3: high-speed steering stabilization applied to the learned V0 racer."""

from __future__ import annotations

from controllers.v0 import Controller as V0Controller
from racing import RobotCommand, RobotSensors

RACING_NAME = "V3 Stable Apex Racer"
RACING_COLOR = "#22c55e"


class Controller:
    def __init__(self) -> None:
        self._base = V0Controller()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        command = self._base(sensors)
        if sensors.contact.wall > 0.0:
            return command
        high_speed = max(0.0, min(1.0, (sensors.odometry.speed_mps - 18.0) / 13.0))
        gain = 1.0 - 0.07 * high_speed
        return RobotCommand(throttle=command.throttle, steer=gain * command.steer)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
