"""V50: aggressive V49 straight target with its jointly trained brake marker."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS

RACING_NAME = "V50 Aggressive Straight Time-Trial Racer"
RACING_COLOR = "#450a0a"
AGGRESSIVE_STRAIGHT_SPEED_MPS = 37.50
AGGRESSIVE_BRAKING_BOUNDARIES_M = (8.0, 48.0, 82.0, 115.0, 140.0, 165.0)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
            straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
