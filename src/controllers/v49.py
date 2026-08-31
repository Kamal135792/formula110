"""V49: V48 with jointly optimized straight speed and braking marker."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS

RACING_NAME = "V49 Trained Straight-Braking Racer"
RACING_COLOR = "#7f1d1d"
STRAIGHT_SPEED_TARGET_MPS = 36.75
STRAIGHT_BRAKING_BOUNDARIES_M = (4.0, 48.0, 82.0, 115.0, 140.0, 165.0)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_boundaries_m=STRAIGHT_BRAKING_BOUNDARIES_M,
            straight_speed_numerator=STRAIGHT_SPEED_TARGET_MPS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
