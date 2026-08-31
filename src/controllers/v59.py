"""V59: V58 plus a trained launch for deep-curve and near-turn starts."""

from __future__ import annotations

from math import isfinite

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS
from controllers.v50 import Controller as NormalStartController
from controllers.v54 import OPTIMIZED_HAZARD_STABILIZATION_TICKS
from controllers.v55 import HAZARD_CURVE_SPEED_GAINS
from controllers.v56 import HAZARD_SPEED_BOUNDARIES_M, HAZARD_STRAIGHT_SPEED_MPS
from controllers.v58 import SENSITIVE_SPEED_BOUNDARIES_M, SENSITIVE_STRAIGHT_SPEED_MPS
from racing import RobotCommand, RobotSensors

RACING_NAME = "V59 Audited Spawn-Conditioned Racer"
RACING_COLOR = "#6366f1"


def _controller(
    *,
    gains: tuple[float, ...],
    speed_mps: float,
    boundaries_m: tuple[float, ...],
    extended: bool = False,
) -> SectorController:
    return SectorController(
        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
        curve_speed_gains=gains,
        speed_boundaries_m=boundaries_m,
        straight_speed_numerator=speed_mps,
        hazard_stable_ticks=OPTIMIZED_HAZARD_STABILIZATION_TICKS,
        extended_start_hazards=extended,
    )


class Controller:
    def __init__(self) -> None:
        self._normal = NormalStartController()
        self._fast_hazard = _controller(
            gains=HAZARD_CURVE_SPEED_GAINS,
            speed_mps=HAZARD_STRAIGHT_SPEED_MPS,
            boundaries_m=HAZARD_SPEED_BOUNDARIES_M,
        )
        self._sensitive_hazard = _controller(
            gains=HAZARD_CURVE_SPEED_GAINS,
            speed_mps=SENSITIVE_STRAIGHT_SPEED_MPS,
            boundaries_m=SENSITIVE_SPEED_BOUNDARIES_M,
        )
        self._extended_hazard = _controller(
            gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_mps=AGGRESSIVE_STRAIGHT_SPEED_MPS,
            boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
            extended=True,
        )
        self._selected: NormalStartController | SectorController = self._normal

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            heading = sensors.imu.heading_degrees
            front = sensors.wall_lidar.front_m
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            extended_curve = 70.0 < heading < 90.0 and far_offset < -7.0
            extended_straight = isfinite(front) and abs(far_offset) < 0.6 and 31.0 < front < 35.0
            straight_hazard = (
                isfinite(front)
                and abs(far_offset) < 0.6
                and (37.0 < front < 40.5 or 42.0 < front < 44.5 or 46.0 < front < 49.5)
            )
            curve_hazard = 80.0 < heading < 95.0 and -5.0 < far_offset < -2.0
            terminal_speed_sensitive = isfinite(front) and 42.0 < front < 44.5 and abs(far_offset) < 0.6
            if extended_curve or extended_straight:
                self._selected = self._extended_hazard
            elif terminal_speed_sensitive:
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
