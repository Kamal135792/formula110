"""V71: V70 with the independently validated faster sector-2 envelope."""

from __future__ import annotations

from controllers.v69 import Controller as TwoStageController
from controllers.v70 import RECOMBINED_INSIDE_OFFSETS_M

RACING_NAME = "V71 Sector-2 Grip-Limit Racer"
RACING_COLOR = "#052e16"
SECTOR_2_LIMIT_GAINS = (1.16, 0.48, 0.34, 0.58, 0.78)


class Controller(TwoStageController):
    def __init__(self) -> None:
        super().__init__(
            recurring_corner_offsets_m=RECOMBINED_INSIDE_OFFSETS_M,
            recurring_curve_speed_gains=SECTOR_2_LIMIT_GAINS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
