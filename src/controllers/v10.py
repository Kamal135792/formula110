"""V10: stable launch followed by the retrained high-speed policy."""

from __future__ import annotations

from controllers.v8 import Controller as StableController
from controllers.v9 import Controller as SprintController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V10 Staged Sprint Racer"
RACING_COLOR = "#f97316"
SPRINT_START_TICK = 480


class Controller:
    def __init__(self) -> None:
        self._stable = StableController()
        self._sprint = SprintController()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        stable_command = self._stable(sensors)
        sprint_command = self._sprint(sensors)
        return stable_command if sensors.tick < SPRINT_START_TICK else sprint_command

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
