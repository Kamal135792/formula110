"""V43: combined partition-trained racing line and corner speed schedule."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M

RACING_NAME = "V43 Partition-Trained Time-Trial Racer"
RACING_COLOR = "#f472b6"
LEARNED_CURVE_SPEED_GAINS = (1.12, 0.78, 0.78, 0.78, 0.78)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=LEARNED_CURVE_SPEED_GAINS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
