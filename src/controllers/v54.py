"""V54: V53 with the hazardous-start stabilization optimized to two seconds."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS

RACING_NAME = "V54 Two-Second Stabilize-Then-Attack Racer"
RACING_COLOR = "#22c55e"
OPTIMIZED_HAZARD_STABILIZATION_TICKS = 120


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
            straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
            hazard_stable_ticks=OPTIMIZED_HAZARD_STABILIZATION_TICKS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
