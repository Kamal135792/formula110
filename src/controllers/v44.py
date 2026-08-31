"""V44: fine-tuned partition speed schedule over V43's learned line."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M

RACING_NAME = "V44 Fine Partition Time-Trial Racer"
RACING_COLOR = "#fb7185"
FINE_CURVE_SPEED_GAINS = (1.16, 0.62, 0.62, 0.62, 0.78)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=FINE_CURVE_SPEED_GAINS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
