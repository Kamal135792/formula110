"""V45: per-sector stability-boundary speed schedule over V44."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M

RACING_NAME = "V45 Sector-Limit Time-Trial Racer"
RACING_COLOR = "#ef4444"
LIMIT_CURVE_SPEED_GAINS = (1.28, 0.48, 0.40, 0.58, 0.78)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=LIMIT_CURVE_SPEED_GAINS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
