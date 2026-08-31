"""V52: jointly evolved partition policy initialized from V51."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController

RACING_NAME = "V52 Jointly Evolved Partition Racer"
RACING_COLOR = "#eab308"
EVOLVED_CORNER_OFFSETS_M = (
    0.9351610479354955,
    1.826485034822548,
    0.8652645732817434,
    1.4075551788063243,
    1.5600494579037036,
)
EVOLVED_CURVE_SPEED_GAINS = (
    1.1759922093069592,
    0.5259409890776665,
    0.42909043933279006,
    0.5829470734537354,
    0.75230349925608,
)
EVOLVED_SPEED_BOUNDARIES_M = (
    3.3383107172286866,
    46.03004751936197,
    80.66866727639056,
    114.79171322726435,
    139.8517350983791,
    164.7251162289935,
)
EVOLVED_STRAIGHT_SPEED_MPS = 38.29713263567378


class Controller(SectorController):
    def __init__(self) -> None:
        super().__init__(
            corner_offsets_m=EVOLVED_CORNER_OFFSETS_M,
            curve_speed_gains=EVOLVED_CURVE_SPEED_GAINS,
            speed_boundaries_m=EVOLVED_SPEED_BOUNDARIES_M,
            straight_speed_numerator=EVOLVED_STRAIGHT_SPEED_MPS,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
