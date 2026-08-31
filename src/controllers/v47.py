"""V47: V46 with its racing line re-optimized at the higher corner speeds."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS

RACING_NAME = "V47 Re-Optimized Sector-Line Racer"
RACING_COLOR = "#b91c1c"
HIGH_SPEED_CORNER_OFFSETS_M = (0.40, 1.70, 0.80, 1.40, 1.80)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=HIGH_SPEED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
