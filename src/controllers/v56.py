"""V56: V55 with a terminal-speed-trained hazard-start straight schedule."""

from __future__ import annotations

from math import isfinite

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v50 import Controller as NormalStartController
from controllers.v54 import OPTIMIZED_HAZARD_STABILIZATION_TICKS
from controllers.v55 import HAZARD_CURVE_SPEED_GAINS
from racing import RobotCommand, RobotSensors

RACING_NAME = "V56 Spawn-Conditioned Terminal-Speed Racer"
RACING_COLOR = "#06b6d4"
HAZARD_STRAIGHT_SPEED_MPS = 46.0
HAZARD_SPEED_BOUNDARIES_M = (3.0, 48.0, 82.0, 115.0, 140.0, 165.0)


class Controller:
    def __init__(self) -> None:
        self._normal = NormalStartController()
        self._hazard = SectorController(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=HAZARD_CURVE_SPEED_GAINS,
            speed_boundaries_m=HAZARD_SPEED_BOUNDARIES_M,
            straight_speed_numerator=HAZARD_STRAIGHT_SPEED_MPS,
            hazard_stable_ticks=OPTIMIZED_HAZARD_STABILIZATION_TICKS,
        )
        self._use_hazard_policy = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            heading = sensors.imu.heading_degrees
            front = sensors.wall_lidar.front_m
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            straight_hazard = (
                isfinite(front)
                and abs(far_offset) < 0.6
                and (37.0 < front < 40.5 or 42.0 < front < 44.5 or 46.0 < front < 49.5)
            )
            curve_hazard = 80.0 < heading < 95.0 and -5.0 < far_offset < -2.0
            self._use_hazard_policy = straight_hazard or curve_hazard
        selected = self._hazard if self._use_hazard_policy else self._normal
        return selected(sensors)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
