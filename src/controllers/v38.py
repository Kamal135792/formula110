"""V38: one-second stable launch followed by track-localized apex control."""

from __future__ import annotations

from controllers.v36 import Controller as PartitionController

RACING_NAME = "V38 One-Second Launch Racer"
RACING_COLOR = "#00ffcc"


class Controller(PartitionController):
    def __init__(self) -> None:
        super().__init__(boost_enabled=False, stable_launch_ticks=60)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
