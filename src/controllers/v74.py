"""V74: V73's wide exit with full throttle from the safe 160 m point."""

from __future__ import annotations

from controllers.v72 import Controller as FullWidthController
from controllers.v73 import EXIT_STRAIGHT_SPEED_MPS

RACING_NAME = "V74 Full-Throttle Wide-Exit Racer"
RACING_COLOR = "#4d7c0f"
FULL_THROTTLE_START_M = 160.0


class Controller(FullWidthController):
    def __init__(self) -> None:
        super().__init__(
            straight_speed_mps=EXIT_STRAIGHT_SPEED_MPS,
            exit_throttle_floor=1.0,
            exit_throttle_start_m=FULL_THROTTLE_START_M,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
