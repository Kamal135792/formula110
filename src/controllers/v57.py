"""V57: route the one terminal-speed-sensitive spawn to a safer hazard branch."""

from __future__ import annotations

from math import isfinite

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS
from controllers.v50 import Controller as NormalStartController
from controllers.v54 import OPTIMIZED_HAZARD_STABILIZATION_TICKS
from controllers.v55 import HAZARD_CURVE_SPEED_GAINS
from controllers.v56 import HAZARD_SPEED_BOUNDARIES_M, HAZARD_STRAIGHT_SPEED_MPS
from racing import RobotCommand, RobotSensors

RACING_NAME = "V57 Three-Branch Spawn-Conditioned Racer"
RACING_COLOR = "#3b82f6"


def _hazard_controller(*, speed_mps: float, boundaries_m: tuple[float, ...]) -> SectorController:
    return SectorController(
        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
        curve_speed_gains=HAZARD_CURVE_SPEED_GAINS,
        speed_boundaries_m=boundaries_m,
        straight_speed_numerator=speed_mps,
        hazard_stable_ticks=OPTIMIZED_HAZARD_STABILIZATION_TICKS,
    )


class Controller:
    def __init__(self) -> None:
        self._normal = NormalStartController()
        self._fast_hazard = _hazard_controller(
            speed_mps=HAZARD_STRAIGHT_SPEED_MPS,
            boundaries_m=HAZARD_SPEED_BOUNDARIES_M,
        )
        self._sensitive_hazard = _hazard_controller(
            speed_mps=AGGRESSIVE_STRAIGHT_SPEED_MPS,
            boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
        )
        self._selected: NormalStartController | SectorController = self._normal

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
            terminal_speed_sensitive = isfinite(front) and 42.0 < front < 44.5 and abs(far_offset) < 0.6
            if terminal_speed_sensitive:
                self._selected = self._sensitive_hazard
            elif straight_hazard or curve_hazard:
                self._selected = self._fast_hazard
            else:
                self._selected = self._normal
        return self._selected(sensors)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
