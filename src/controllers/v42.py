"""V42: combined line learned by V41's per-sector coordinate search."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController

RACING_NAME = "V42 Learned Sector-Line Racer"
RACING_COLOR = "#a855f7"
LEARNED_CORNER_OFFSETS_M = (0.8, 1.8, 0.8, 1.4, 1.8)


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(corner_offsets_m=LEARNED_CORNER_OFFSETS_M)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
