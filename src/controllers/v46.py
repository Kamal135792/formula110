"""V46: V45 speed with the rare sector-0/1 instability removed."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M

RACING_NAME = "V46 Robust Sector-Limit Racer"
RACING_COLOR = "#dc2626"
ROBUST_LIMIT_CURVE_SPEED_GAINS = (1.16, 0.48, 0.40, 0.58, 0.78)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
