"""V63: V62 plus a 90-tick launch for near-turn straight starts."""

from __future__ import annotations

from math import isfinite

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS
from controllers.v62 import Controller as DualHairpinController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V63 Fully Launch-Partitioned Racer"
RACING_COLOR = "#f43f5e"
NEAR_TURN_STABILIZATION_TICKS = 90


class Controller(DualHairpinController):
    def __init__(self) -> None:
        super().__init__()
        self._near_turn = SectorController(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
            straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
            hazard_stable_ticks=NEAR_TURN_STABILIZATION_TICKS,
            extended_start_hazards=True,
        )

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            front = sensors.wall_lidar.front_m
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            if isfinite(front) and abs(far_offset) < 0.6 and 31.0 < front < 35.0:
                self._selected = self._near_turn
                return self._selected(sensors)
        return super().__call__(sensors)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
