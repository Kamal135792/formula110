"""V48: V46 with partition handoff positions optimized separately."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS

RACING_NAME = "V48 Optimized-Handoff Sector Racer"
RACING_COLOR = "#991b1b"
OPTIMIZED_SPEED_BOUNDARIES_M = (12.0, 48.0, 82.0, 115.0, 140.0, 165.0)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_boundaries_m=OPTIMIZED_SPEED_BOUNDARIES_M,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
