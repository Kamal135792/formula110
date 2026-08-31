"""V73: V72's full-width final corner with a faster post-exit straight target."""

from __future__ import annotations

from controllers.v72 import Controller as FullWidthController

RACING_NAME = "V73 Full-Width Exit-Speed Racer"
RACING_COLOR = "#65a30d"
EXIT_STRAIGHT_SPEED_MPS = 39.0


class Controller(FullWidthController):
    def __init__(self) -> None:
        super().__init__(straight_speed_mps=EXIT_STRAIGHT_SPEED_MPS)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
