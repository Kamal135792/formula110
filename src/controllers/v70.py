"""V70: V69 launch safety with a recombined high-speed inside lap line."""

from __future__ import annotations

from controllers.v69 import Controller as TwoStageController

RACING_NAME = "V70 Recombined Inside-Line Racer"
RACING_COLOR = "#14532d"
RECOMBINED_INSIDE_OFFSETS_M = (1.2, 2.2, 1.2, 1.8, 2.2)


class Controller(TwoStageController):
    def __init__(self) -> None:
        super().__init__(recurring_corner_offsets_m=RECOMBINED_INSIDE_OFFSETS_M)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
