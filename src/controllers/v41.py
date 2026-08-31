"""V41: track-localized, independently tunable racing-line sectors."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from math import isfinite

from controllers.track_localizer import TrackLocalizer
from controllers.v0 import PlannerConfig
from controllers.v24 import Controller as BaseController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V41 Sector Racing-Line Racer"
RACING_COLOR = "#c084fc"

SECTOR_BOUNDARIES_M = (17.0, 48.0, 80.0, 110.0, 140.0, 165.0)
CORNER_OFFSETS_M = (1.4, 1.4, 1.4, 1.4, 1.4)
CURVE_SPEED_GAINS = (0.972, 0.972, 0.972, 0.972, 0.972)
LINE_STEER_GAIN = 0.059


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _corner_sector(progress_m: float) -> int | None:
    if progress_m < 17.0:
        return None
    if 17.0 <= progress_m < 48.0:
        return 0
    if progress_m < 80.0:
        return 1
    if progress_m < 110.0:
        return 2
    if progress_m < 140.0:
        return 3
    if progress_m < 165.0:
        return 4
    return None


def _scheduled_sector(progress_m: float, boundaries_m: tuple[float, ...]) -> int | None:
    if progress_m < boundaries_m[0] or progress_m >= boundaries_m[-1]:
        return None
    for sector, end_m in enumerate(boundaries_m[1:]):
        if progress_m < end_m:
            return sector
    return None


class Controller:
    def __init__(
        self,
        *,
        corner_offsets_m: tuple[float, ...] = CORNER_OFFSETS_M,
        curve_speed_gains: tuple[float, ...] = CURVE_SPEED_GAINS,
        speed_boundaries_m: tuple[float, ...] = SECTOR_BOUNDARIES_M,
        straight_speed_numerator: float = 36.0,
        initial_stable_ticks: int = 60,
        hazard_stable_ticks: int | None = None,
        extended_start_hazards: bool = False,
    ) -> None:
        if len(corner_offsets_m) != 5:
            raise ValueError("corner_offsets_m must contain five values")
        if len(curve_speed_gains) != 5:
            raise ValueError("curve_speed_gains must contain five values")
        if len(speed_boundaries_m) != 6 or any(
            end_m <= start_m for start_m, end_m in pairwise(speed_boundaries_m)
        ):
            raise ValueError("speed_boundaries_m must contain six increasing values")
        self._corner_offsets_m = corner_offsets_m
        self._curve_speed_gains = curve_speed_gains
        self._speed_boundaries_m = speed_boundaries_m
        self._straight_speed_numerator = straight_speed_numerator
        self._initial_stable_ticks = initial_stable_ticks
        self._hazard_stable_ticks = hazard_stable_ticks
        self._extended_start_hazards = extended_start_hazards
        self._localizer = TrackLocalizer()
        self._stable = BaseController()
        self._base = BaseController()
        default_config = self._base.planner_config
        self._straight_config: PlannerConfig = replace(
            default_config,
            speed_numerator=self._straight_speed_numerator,
        )
        self._sector_configs = tuple(
            replace(default_config, curve_speed_gain=gain) for gain in self._curve_speed_gains
        )
        self._use_stable = False
        self._unwrapped_progress_m = 0.0
        self._last_progress_m = 0.0
        self._last_odometry_m = 0.0
        self._next_crossings_m: list[float] = []
        self.sector_crossings: list[tuple[float, int]] = []

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        self._record_sector_crossings(progress, sensors)
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
            extended_curve_hazard = (
                self._extended_start_hazards and 40.0 < heading < 90.0 and far_offset < -7.0
            )
            extended_straight_hazard = (
                self._extended_start_hazards
                and isfinite(front)
                and abs(far_offset) < 0.6
                and 31.0 < front < 35.0
            )
            self._use_stable = (
                straight_hazard or curve_hazard or extended_curve_hazard or extended_straight_hazard
            )

        stable = self._stable(sensors)
        sector = _corner_sector(progress)
        speed_sector = _scheduled_sector(progress, self._speed_boundaries_m)
        self._base.planner_config = (
            self._straight_config if speed_sector is None else self._sector_configs[speed_sector]
        )
        command = self._base(sensors)
        stabilizing_hazard = self._use_stable and (
            self._hazard_stable_ticks is None or sensors.tick < self._hazard_stable_ticks
        )
        if stabilizing_hazard or sensors.tick < self._initial_stable_ticks:
            return stable
        if sensors.contact.wall > 0.0 or not sensors.camera.visible:
            return command

        if sector is None:
            return command
        offsets = (*sensors.camera.lookahead_offsets_m, 0.0, 0.0, 0.0)
        near_curve = _clamp(abs(offsets[0]) / 2.2, 0.0, 1.0)
        apex = _clamp(near_curve * 1.8, 0.0, 1.0)
        direction_source = offsets[2] if abs(offsets[2]) > abs(offsets[0]) else offsets[0]
        direction = 1.0 if direction_source > 0.0 else -1.0 if direction_source < 0.0 else 0.0
        desired_car_offset = direction * self._corner_offsets_m[sector] * apex
        correction = _clamp(LINE_STEER_GAIN * desired_car_offset, -0.08, 0.08)
        return RobotCommand(throttle=command.throttle, steer=_clamp(command.steer + correction, -1.0, 1.0))

    def _record_sector_crossings(self, progress_m: float, sensors: RobotSensors) -> None:
        total = self._localizer.total_length_m
        if sensors.tick == 0:
            self._last_progress_m = progress_m
            self._last_odometry_m = sensors.odometry.distance_m
            self._unwrapped_progress_m = progress_m
            self._next_crossings_m = [
                boundary if boundary > progress_m else boundary + total for boundary in SECTOR_BOUNDARIES_M
            ]
            self.sector_crossings.clear()
            return
        delta = max(0.0, sensors.odometry.distance_m - self._last_odometry_m)
        self._last_odometry_m = sensors.odometry.distance_m
        self._last_progress_m = progress_m
        self._unwrapped_progress_m += delta
        for index, next_crossing in enumerate(self._next_crossings_m):
            if self._unwrapped_progress_m >= next_crossing:
                self.sector_crossings.append((SECTOR_BOUNDARIES_M[index], sensors.tick))
                self._next_crossings_m[index] += total

    def copy_for_car(self) -> Controller:
        return Controller(
            corner_offsets_m=self._corner_offsets_m,
            curve_speed_gains=self._curve_speed_gains,
            speed_boundaries_m=self._speed_boundaries_m,
            straight_speed_numerator=self._straight_speed_numerator,
            initial_stable_ticks=self._initial_stable_ticks,
            hazard_stable_ticks=self._hazard_stable_ticks,
            extended_start_hazards=self._extended_start_hazards,
        )


def create_controller() -> Controller:
    return Controller()
