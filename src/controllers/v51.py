"""V51: high straight target with an early, low-runoff braking handoff."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS

RACING_NAME = "V51 High-Speed Early-Brake Racer"
RACING_COLOR = "#f97316"
HIGH_STRAIGHT_SPEED_MPS = 38.25
EARLY_BRAKING_BOUNDARIES_M = (4.0, 48.0, 82.0, 115.0, 140.0, 165.0)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_boundaries_m=EARLY_BRAKING_BOUNDARIES_M,
            straight_speed_numerator=HIGH_STRAIGHT_SPEED_MPS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
