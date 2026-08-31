"""V67: repeat the selected safe corner path on every lap."""

from __future__ import annotations

from controllers.v66 import Controller as PathMixtureController

RACING_NAME = "V67 Repeating Zero-Damage Path Racer"
RACING_COLOR = "#047857"


class Controller(PathMixtureController):
    def __init__(self) -> None:
        super().__init__(repeat_guard=True)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
