"""V68: repeat high-speed-safe paths only for delayed-failure spawn classes."""

from __future__ import annotations

from controllers.v66 import Controller as PathMixtureController

RACING_NAME = "V68 Selective Recurring Path Racer"
RACING_COLOR = "#065f46"


class Controller(PathMixtureController):
    def __init__(self) -> None:
        super().__init__(selective_repeat=True)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
