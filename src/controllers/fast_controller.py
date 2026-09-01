from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum, auto
from itertools import pairwise
from math import atan, atan2, fmod, inf, isfinite, radians, tanh
from pathlib import Path
from typing import cast

from racing import RobotCommand, RobotSensors

RACING_NAME = "Fast Hybrid Racer"
RACING_COLOR = "#3f6212"

_MAP_PATH = Path(__file__).with_name("track_signature_map.json")
_DECISION_INTERVAL_TICKS = 3
_ACTION_RESIDUAL_SCALES = (0.30, 0.25)
_WHEELBASE_M = 1.40
_MAX_STEERING_RADIANS = radians(25.0)

_SECTOR_BOUNDARIES_M = (17.0, 48.0, 80.0, 110.0, 140.0, 165.0)
_LEARNED_CORNER_OFFSETS_M = (0.8, 1.8, 0.8, 1.4, 1.8)
_RECOMBINED_INSIDE_OFFSETS_M = (1.2, 2.2, 1.2, 1.8, 2.2)
_ROBUST_LIMIT_CURVE_SPEED_GAINS = (1.16, 0.48, 0.40, 0.58, 0.78)
_SECTOR_2_LIMIT_GAINS = (1.16, 0.48, 0.34, 0.58, 0.78)
_HAZARD_CURVE_SPEED_GAINS = (1.16, 0.48, 0.32, 0.58, 0.82)
_AGGRESSIVE_BRAKING_BOUNDARIES_M = (8.0, 48.0, 82.0, 115.0, 140.0, 165.0)
_HAZARD_SPEED_BOUNDARIES_M = (3.0, 48.0, 82.0, 115.0, 140.0, 165.0)
_SENSITIVE_SPEED_BOUNDARIES_M = (2.0, 48.0, 82.0, 115.0, 140.0, 165.0)
_AGGRESSIVE_STRAIGHT_SPEED_MPS = 37.5
_HAZARD_STRAIGHT_SPEED_MPS = 46.0
_SENSITIVE_STRAIGHT_SPEED_MPS = 44.0
_LINE_STEER_GAIN = 0.059

_ACTOR_WEIGHTS = (
    (
        -0.012891643461745428,
        -0.006308734647885359,
        0.0013658898769210019,
        0.002359783334640391,
        0.005064361884900923,
        0.00501141505060541,
        0.004560651639535458,
        0.003020085842494152,
        0.001637975831663182,
        -0.006423270508484834,
        -0.0017846577285371423,
        -0.004523763790207662,
        0.004962894092629445,
        0.004685837780562896,
        -0.007691971585064725,
    ),
    (
        -0.010506111245334088,
        -0.005911195113946478,
        -0.0006400749312593041,
        -0.00044169366516628,
        -0.0007485190573554287,
        0.00044854064703459223,
        0.002026216326105548,
        0.001939369765089602,
        0.0020751858232405614,
        -0.0027421117306587533,
        0.0010340378957745928,
        -0.003643364369297461,
        -0.0005051992688797285,
        -0.0008915622387823454,
        -0.003741887010489864,
    ),
)

_Path = tuple[float, float, float, float, float]
_EARLY_PATH: _Path = (35.0, 41.0, 48.0, 48.0, -0.20)
_SPAWN_24_PATH: _Path = (23.9851, 40.5777, 45.3321, 48.6888, -0.3965)
_SPAWN_00_PATH: _Path = (21.0904, 44.0201, 52.0517, 55.1612, -0.2370)
_SPAWN_11_PATH: _Path = (23.3904, 37.4152, 52.4320, 57.9421, -0.1821)
_SPAWN_21_PATH: _Path = (37.7245, 46.0399, 51.2473, 52.1685, -0.1172)
_SPAWN_34_PATH: _Path = (20.5823, 34.2384, 48.9733, 54.5777, -0.3382)
_SPAWN_36_PATH: _Path = (25.6669, 42.5031, 51.9515, 55.2365, -0.3033)
_SPAWN_37_PATH: _Path = (38.5395, 38.7703, 50.6104, 54.3517, -0.2783)
_SPAWN_40_DEEP_PATH: _Path = (23.8720, 39.9245, 50.0656, 57.4937, -0.3062)
_SPAWN_40_SHALLOW_PATH: _Path = (21.5671, 40.8609, 50.6813, 54.5109, -0.1104)
_SPAWN_179_PATH: _Path = (23.1085, 36.7734, 50.3737, 52.6551, -0.0975)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _angle_error_degrees(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def _wrapped_distance(first: float, second: float, total: float) -> float:
    return abs((first - second + total / 2.0) % total - total / 2.0)


def _triangle(progress_m: float, start_m: float, peak_m: float, end_m: float) -> float:
    if progress_m <= start_m or progress_m >= end_m:
        return 0.0
    if progress_m <= peak_m:
        return (progress_m - start_m) / (peak_m - start_m)
    return (end_m - progress_m) / (end_m - peak_m)


class _TrackLocalizer:
    """Estimate centerline progress from the fixed public-sensor signature map."""

    def __init__(self) -> None:
        record = cast(dict[str, object], json.loads(_MAP_PATH.read_text(encoding="utf-8")))
        self.total_length_m = float(cast(float, record["total_length_m"]))
        rows = cast(list[list[float]], record["samples"])
        self._samples = tuple(tuple(float(value) for value in row) for row in rows)
        self.progress_m: float | None = None
        self._last_odometry_m = 0.0

    def update(self, sensors: RobotSensors) -> float:
        if sensors.tick == 0:
            self.progress_m = None
            self._last_odometry_m = sensors.odometry.distance_m

        odometry_delta = max(0.0, sensors.odometry.distance_m - self._last_odometry_m)
        self._last_odometry_m = sensors.odometry.distance_m
        predicted = None if self.progress_m is None else (self.progress_m + odometry_delta) % self.total_length_m
        desired_heading = sensors.imu.heading_degrees + sensors.camera.heading_error_degrees
        best_progress = self.locate_signature(
            desired_heading_degrees=desired_heading,
            center_offset_m=sensors.camera.center_offset_m,
            lookahead_offsets_m=sensors.camera.lookahead_offsets_m,
            predicted_progress_m=predicted,
        )
        self.progress_m = fmod(best_progress, self.total_length_m)
        return self.progress_m

    def locate_signature(
        self,
        *,
        desired_heading_degrees: float,
        center_offset_m: float,
        lookahead_offsets_m: tuple[float, ...],
        predicted_progress_m: float | None = None,
    ) -> float:
        offsets = tuple(value - center_offset_m for value in lookahead_offsets_m)
        padded = (*offsets, 0.0, 0.0, 0.0)
        best_progress = 0.0
        best_error = float("inf")
        for row in self._samples:
            progress, heading, near, middle, far = row
            if (
                predicted_progress_m is not None
                and _wrapped_distance(progress, predicted_progress_m, self.total_length_m) > 10.0
            ):
                continue
            error = (
                (_angle_error_degrees(desired_heading_degrees, heading) / 10.0) ** 2
                + ((padded[0] - near) / 1.5) ** 2
                + ((padded[1] - middle) / 3.5) ** 2
                + ((padded[2] - far) / 6.0) ** 2
            )
            if predicted_progress_m is not None:
                error += (_wrapped_distance(progress, predicted_progress_m, self.total_length_m) / 4.0) ** 2
            if error < best_error:
                best_error = error
                best_progress = progress
        return best_progress


@dataclass(frozen=True, slots=True)
class _PlannerConfig:
    pure_pursuit_gain: float = 1.65
    heading_gain: float = 0.34
    center_gain: float = 0.12
    speed_numerator: float = 36.0
    curve_speed_gain: float = 0.972
    minimum_corner_speed: float = 12.0
    apex_bias_m: float = 0.0


_BASE_PLANNER_CONFIG = _PlannerConfig()


def _dot(weights: tuple[float, ...], features: tuple[float, ...]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features, strict=True))


class _ResidualGeometricPlanner:
    """V24's inference path: geometric planner plus the V21 actor residual."""

    def __init__(self) -> None:
        self.planner_config = _BASE_PLANNER_CONFIG
        self._last_command = RobotCommand()
        self._held_command = RobotCommand()
        self._smoothed_target_speed_mps: float | None = None
        self._brake_pulse_active = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            self._last_command = RobotCommand()
            self._held_command = RobotCommand()
            self._smoothed_target_speed_mps = None
            self._brake_pulse_active = False

        if sensors.tick % _DECISION_INTERVAL_TICKS != 0:
            return self._held_command

        base_command, self._smoothed_target_speed_mps = _geometric_command(
            sensors,
            previous_target_speed_mps=self._smoothed_target_speed_mps,
            config=self.planner_config,
        )
        features = _policy_features(sensors, base_command, self._last_command)
        throttle, steer = _policy_action_means(features, base_command)
        if base_command.throttle <= 0.0:
            throttle = min(throttle, base_command.throttle)

        if sensors.contact.wall <= 0.0:
            if throttle < 0.0 and not self._brake_pulse_active:
                throttle = max(-1.0, throttle)
                self._brake_pulse_active = True
            else:
                if self._brake_pulse_active:
                    throttle = 0.0
                self._brake_pulse_active = False

        command = RobotCommand(
            throttle=_clamp(throttle, -1.0, 1.0),
            steer=_clamp(steer, -1.0, 1.0),
        )
        self._last_command = command
        self._held_command = command
        return command


def _policy_action_means(
    features: tuple[float, ...],
    base_command: RobotCommand,
) -> tuple[float, float]:
    throttle_residual = _ACTION_RESIDUAL_SCALES[0] * tanh(_dot(_ACTOR_WEIGHTS[0], features))
    steer_residual = _ACTION_RESIDUAL_SCALES[1] * tanh(_dot(_ACTOR_WEIGHTS[1], features))
    return (base_command.throttle + throttle_residual, base_command.steer + steer_residual)


def _policy_features(
    sensors: RobotSensors,
    base_command: RobotCommand,
    previous_command: RobotCommand,
) -> tuple[float, ...]:
    lookahead_offsets = _camera_lookahead_offsets(sensors)
    left_wall = _capped_distance(sensors.wall_lidar.left_m, 20.0)
    right_wall = _capped_distance(sensors.wall_lidar.right_m, 20.0)
    front_wall = _capped_distance(sensors.wall_lidar.front_m, 25.0)
    curve_severity = _curve_severity(sensors, lookahead_offsets)
    return (
        1.0,
        _clamp(sensors.odometry.speed_mps / 16.0, -1.0, 1.0),
        _clamp(sensors.camera.center_offset_m / 3.0, -1.5, 1.5),
        _clamp(sensors.camera.heading_error_degrees / 60.0, -1.5, 1.5),
        _clamp(lookahead_offsets[0] / 4.0, -1.5, 1.5),
        _clamp(lookahead_offsets[1] / 9.0, -1.5, 1.5),
        _clamp(lookahead_offsets[2] / 16.0, -1.5, 1.5),
        _clamp(sensors.imu.yaw_rate_degrees_per_s / 140.0, -1.5, 1.5),
        _clamp(sensors.imu.lateral_acceleration_mps2 / 45.0, -1.5, 1.5),
        1.0 - front_wall / 25.0,
        _clamp((left_wall - right_wall) / 20.0, -1.0, 1.0),
        previous_command.throttle,
        previous_command.steer,
        base_command.steer,
        curve_severity,
    )


def _geometric_command(
    sensors: RobotSensors,
    *,
    previous_target_speed_mps: float | None,
    config: _PlannerConfig,
) -> tuple[RobotCommand, float | None]:
    speed = max(0.0, sensors.odometry.speed_mps)
    if sensors.contact.wall > 0.0:
        open_side_steer = -0.72 if sensors.lidar.left_m > sensors.lidar.right_m else 0.72
        return RobotCommand(throttle=-1.0 if speed > 0.5 else -0.42, steer=open_side_steer), None
    if not sensors.camera.visible:
        return RobotCommand(throttle=0.18, steer=0.0), previous_target_speed_mps

    lookahead_offsets = _camera_lookahead_offsets(sensors)
    lookahead_distance = _clamp(4.0 + speed * 0.72, 4.0, 16.0)
    target_offset = _interpolated_lookahead_offset(lookahead_distance, lookahead_offsets)
    far_curve = _clamp(lookahead_offsets[2] / 8.0, -1.0, 1.0)
    target_offset += config.apex_bias_m * far_curve
    target_angle = atan2(target_offset, lookahead_distance)
    steering_angle = atan(2.0 * _WHEELBASE_M * target_angle / max(2.0, lookahead_distance))
    pure_pursuit_steer = steering_angle / _MAX_STEERING_RADIANS
    heading_term = sensors.camera.heading_error_degrees / 90.0
    center_term = sensors.camera.center_offset_m / 4.0
    yaw_damping = sensors.imu.yaw_rate_degrees_per_s / 1_100.0
    steer = _clamp(
        config.pure_pursuit_gain * pure_pursuit_steer
        + config.heading_gain * heading_term
        + config.center_gain * center_term
        - yaw_damping,
        -1.0,
        1.0,
    )

    curve_severity = _curve_severity(sensors, lookahead_offsets)
    raw_target_speed = _clamp(
        config.speed_numerator / (1.0 + config.curve_speed_gain * curve_severity),
        config.minimum_corner_speed,
        config.speed_numerator,
    )
    tracking_error = max(
        abs(sensors.camera.center_offset_m) / 2.4,
        abs(sensors.camera.heading_error_degrees) / 55.0,
    )
    if tracking_error > 0.55:
        raw_target_speed /= 1.0 + 0.38 * (tracking_error - 0.55) ** 2
        raw_target_speed = max(10.5, raw_target_speed)

    target_speed = raw_target_speed
    if previous_target_speed_mps is not None:
        target_speed = _clamp(
            raw_target_speed,
            previous_target_speed_mps - 3.0,
            previous_target_speed_mps + 1.5,
        )
    speed_error = target_speed - speed
    if speed_error < -0.8:
        desired_throttle = _clamp(0.30 * speed_error, -1.0, 0.0)
    elif speed_error > 0.8:
        desired_throttle = _clamp(0.28 + 0.13 * speed_error, 0.0, 1.0)
    else:
        desired_throttle = _clamp(0.08 + 0.08 * speed_error, -0.20, 0.24)
    return RobotCommand(throttle=desired_throttle, steer=steer), target_speed


def _camera_lookahead_offsets(sensors: RobotSensors) -> tuple[float, float, float]:
    values = sensors.camera.lookahead_offsets_m
    if len(values) >= 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    padded = [float(value) for value in values]
    while len(padded) < 3:
        padded.append(padded[-1] if padded else sensors.camera.center_offset_m)
    return (padded[0], padded[1], padded[2])


def _interpolated_lookahead_offset(distance_m: float, offsets: tuple[float, float, float]) -> float:
    if distance_m <= 9.0:
        weight = (distance_m - 4.0) / 5.0
        return offsets[0] + _clamp(weight, 0.0, 1.0) * (offsets[1] - offsets[0])
    weight = (distance_m - 9.0) / 7.0
    return offsets[1] + _clamp(weight, 0.0, 1.0) * (offsets[2] - offsets[1])


def _curve_severity(sensors: RobotSensors, offsets: tuple[float, float, float]) -> float:
    return _clamp(
        max(
            abs(sensors.camera.heading_error_degrees) / 55.0,
            0.62 * abs(offsets[0]) / 4.0,
            0.90 * abs(offsets[1]) / 9.0,
            1.25 * abs(offsets[2]) / 16.0,
        ),
        0.0,
        1.5,
    )


def _capped_distance(distance_m: float, cap_m: float) -> float:
    if distance_m == inf or not isfinite(distance_m):
        return cap_m
    return _clamp(distance_m, 0.0, cap_m)


def _corner_sector(progress_m: float) -> int | None:
    if progress_m < 17.0:
        return None
    if progress_m < 48.0:
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


class _SectorController:
    """V41's localized racing line and speed schedule."""

    def __init__(
        self,
        *,
        corner_offsets_m: tuple[float, ...],
        curve_speed_gains: tuple[float, ...],
        speed_boundaries_m: tuple[float, ...],
        straight_speed_numerator: float,
        initial_stable_ticks: int = 60,
        hazard_stable_ticks: int | None = None,
        extended_start_hazards: bool = False,
    ) -> None:
        if len(corner_offsets_m) != 5 or len(curve_speed_gains) != 5:
            raise ValueError("sector schedules must contain five values")
        if len(speed_boundaries_m) != 6 or any(end_m <= start_m for start_m, end_m in pairwise(speed_boundaries_m)):
            raise ValueError("speed boundaries must contain six increasing values")
        self._corner_offsets_m = corner_offsets_m
        self._speed_boundaries_m = speed_boundaries_m
        self._initial_stable_ticks = initial_stable_ticks
        self._hazard_stable_ticks = hazard_stable_ticks
        self._extended_start_hazards = extended_start_hazards
        self._stable = _ResidualGeometricPlanner()
        self._base = _ResidualGeometricPlanner()
        default_config = self._base.planner_config
        self._straight_config = replace(default_config, speed_numerator=straight_speed_numerator)
        self._sector_configs = tuple(replace(default_config, curve_speed_gain=gain) for gain in curve_speed_gains)
        self._use_stable = False

    def __call__(self, sensors: RobotSensors, progress: float) -> RobotCommand:
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
            extended_curve_hazard = self._extended_start_hazards and 40.0 < heading < 90.0 and far_offset < -7.0
            extended_straight_hazard = (
                self._extended_start_hazards and isfinite(front) and abs(far_offset) < 0.6 and 31.0 < front < 35.0
            )
            self._use_stable = straight_hazard or curve_hazard or extended_curve_hazard or extended_straight_hazard

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
        if sensors.contact.wall > 0.0 or not sensors.camera.visible or sector is None:
            return command

        offsets = (*sensors.camera.lookahead_offsets_m, 0.0, 0.0, 0.0)
        near_curve = _clamp(abs(offsets[0]) / 2.2, 0.0, 1.0)
        apex = _clamp(near_curve * 1.8, 0.0, 1.0)
        direction_source = offsets[2] if abs(offsets[2]) > abs(offsets[0]) else offsets[0]
        direction = 1.0 if direction_source > 0.0 else -1.0 if direction_source < 0.0 else 0.0
        desired_car_offset = direction * self._corner_offsets_m[sector] * apex
        correction = _clamp(_LINE_STEER_GAIN * desired_car_offset, -0.08, 0.08)
        return RobotCommand(
            throttle=command.throttle,
            steer=_clamp(command.steer + correction, -1.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class _LaunchPolicy:
    gains: tuple[float, ...]
    speed_mps: float
    boundaries_m: tuple[float, ...]
    hazard_ticks: int | None
    extended: bool = False


_NORMAL_POLICY = _LaunchPolicy(
    _ROBUST_LIMIT_CURVE_SPEED_GAINS,
    _AGGRESSIVE_STRAIGHT_SPEED_MPS,
    _AGGRESSIVE_BRAKING_BOUNDARIES_M,
    None,
)
_FAST_HAZARD_POLICY = _LaunchPolicy(
    _HAZARD_CURVE_SPEED_GAINS,
    _HAZARD_STRAIGHT_SPEED_MPS,
    _HAZARD_SPEED_BOUNDARIES_M,
    120,
)
_SENSITIVE_HAZARD_POLICY = _LaunchPolicy(
    _HAZARD_CURVE_SPEED_GAINS,
    _SENSITIVE_STRAIGHT_SPEED_MPS,
    _SENSITIVE_SPEED_BOUNDARIES_M,
    120,
)
_EXTENDED_HAZARD_POLICY = _LaunchPolicy(
    _ROBUST_LIMIT_CURVE_SPEED_GAINS,
    _AGGRESSIVE_STRAIGHT_SPEED_MPS,
    _AGGRESSIVE_BRAKING_BOUNDARIES_M,
    120,
    True,
)
_NINETY_TICK_HAZARD_POLICY = _LaunchPolicy(
    _ROBUST_LIMIT_CURVE_SPEED_GAINS,
    _AGGRESSIVE_STRAIGHT_SPEED_MPS,
    _AGGRESSIVE_BRAKING_BOUNDARIES_M,
    90,
    True,
)


def _new_sector_controller(policy: _LaunchPolicy) -> _SectorController:
    return _SectorController(
        corner_offsets_m=_LEARNED_CORNER_OFFSETS_M,
        curve_speed_gains=policy.gains,
        speed_boundaries_m=policy.boundaries_m,
        straight_speed_numerator=policy.speed_mps,
        hazard_stable_ticks=policy.hazard_ticks,
        extended_start_hazards=policy.extended,
    )


def _select_launch_policy(sensors: RobotSensors) -> _LaunchPolicy:
    heading = sensors.imu.heading_degrees
    front = sensors.wall_lidar.front_m
    far_offset = sensors.camera.lookahead_offsets_m[-1]
    if isfinite(front) and abs(far_offset) < 0.6 and 31.0 < front < 35.0:
        return _NINETY_TICK_HAZARD_POLICY
    if 40.0 < heading < 50.0 and far_offset < -7.0:
        return _NINETY_TICK_HAZARD_POLICY
    extended_curve = 50.0 < heading < 90.0 and far_offset < -7.0
    straight_hazard = (
        isfinite(front)
        and abs(far_offset) < 0.6
        and (37.0 < front < 40.5 or 42.0 < front < 44.5 or 46.0 < front < 49.5)
    )
    curve_hazard = 80.0 < heading < 95.0 and -5.0 < far_offset < -2.0
    terminal_speed_sensitive = isfinite(front) and 42.0 < front < 44.5 and abs(far_offset) < 0.6
    if extended_curve:
        return _EXTENDED_HAZARD_POLICY
    if terminal_speed_sensitive:
        return _SENSITIVE_HAZARD_POLICY
    if straight_hazard or curve_hazard:
        return _FAST_HAZARD_POLICY
    return _NORMAL_POLICY


class _GuardedLaunchController:
    def __init__(self, path: _Path, policy: _SectorController) -> None:
        self._policy = policy
        (
            self._guard_start_m,
            self._guard_peak_start_m,
            self._guard_peak_end_m,
            self._guard_end_m,
            self._avoidance_steer,
        ) = path
        self._guard_active = True
        self._started_beyond_guard = False

    def __call__(self, sensors: RobotSensors, progress: float) -> RobotCommand:
        if sensors.tick == 0:
            self._guard_active = True
            self._started_beyond_guard = progress >= self._guard_end_m

        command = self._policy(sensors, progress)
        if not self._guard_active:
            return command
        if self._started_beyond_guard:
            if progress < self._guard_start_m:
                self._started_beyond_guard = False
            else:
                return command
        if progress >= self._guard_end_m:
            self._guard_active = False
            return command
        if progress < self._guard_start_m:
            return command
        if progress < self._guard_peak_start_m:
            weight = (progress - self._guard_start_m) / (self._guard_peak_start_m - self._guard_start_m)
        elif progress <= self._guard_peak_end_m:
            weight = 1.0
        else:
            weight = (self._guard_end_m - progress) / (self._guard_end_m - self._guard_peak_end_m)
        steer = _clamp(
            command.steer + self._avoidance_steer * _clamp(weight, 0.0, 1.0),
            -1.0,
            1.0,
        )
        return RobotCommand(throttle=command.throttle, steer=steer)


class _FirstCornerLaunchController:
    """Flattened V63 classifier, V66 spawn path, and V64 first-corner guard."""

    def __init__(self) -> None:
        self._selected: _GuardedLaunchController | None = None

    def __call__(self, sensors: RobotSensors, progress: float) -> RobotCommand:
        if self._selected is None:
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            if progress < 1.0:
                path = _SPAWN_00_PATH
            elif progress >= 175.0:
                path = _SPAWN_179_PATH
            elif 10.5 <= progress < 11.5:
                path = _SPAWN_11_PATH
            elif 20.5 <= progress < 22.5:
                path = _SPAWN_21_PATH
            elif 23.0 <= progress < 26.0:
                path = _SPAWN_24_PATH
            elif 26.5 <= progress < 28.5 or 32.5 <= progress < 35.0:
                path = _SPAWN_34_PATH
            elif 35.0 <= progress < 36.5:
                path = _SPAWN_36_PATH
            elif 36.5 <= progress < 38.5:
                path = _SPAWN_37_PATH
            elif 39.0 <= progress < 41.5:
                path = _SPAWN_40_DEEP_PATH if far_offset < -11.25 else _SPAWN_40_SHALLOW_PATH
            else:
                path = _EARLY_PATH
            policy = _select_launch_policy(sensors)
            self._selected = _GuardedLaunchController(path, _new_sector_controller(policy))
        return self._selected(sensors, progress)


class _RecurringLapController:
    """V75's recurring sector policy plus final-corner correction."""

    def __init__(self) -> None:
        self._localizer = _TrackLocalizer()
        gains = (*_SECTOR_2_LIMIT_GAINS[:4], 0.78)
        self._sector = _SectorController(
            corner_offsets_m=_RECOMBINED_INSIDE_OFFSETS_M,
            curve_speed_gains=gains,
            speed_boundaries_m=_AGGRESSIVE_BRAKING_BOUNDARIES_M,
            straight_speed_numerator=39.0,
        )

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        command = self._sector(sensors, progress)
        correction = 0.04 * _triangle(progress, 132.0, 147.0, 155.0)
        correction += -0.09 * _triangle(progress, 148.0, 160.0, 170.0)
        throttle = command.throttle
        if 157.0 <= progress <= 165.0:
            throttle = max(throttle, 1.0)
        return RobotCommand(
            throttle=throttle,
            steer=_clamp(command.steer + correction, -1.0, 1.0),
        )


class _RaceState(Enum):
    LAUNCH = auto()
    RECURRING = auto()


class Controller:
    """Explicit launch-to-recurring state machine for one car and race."""

    def __init__(self) -> None:
        self._state = _RaceState.LAUNCH
        self._launch = _FirstCornerLaunchController()
        self._recurring: _RecurringLapController | None = None
        self._localizer = _TrackLocalizer()
        self._started_beyond_merge = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if self._state is _RaceState.RECURRING:
            if self._recurring is None:
                raise RuntimeError("recurring state requires a recurring controller")
            return self._recurring(sensors)

        progress = self._localizer.update(sensors)
        if sensors.tick == 0:
            self._started_beyond_merge = progress >= 60.0
        if self._started_beyond_merge:
            if progress < 10.0:
                self._started_beyond_merge = False
        elif progress >= 60.0:
            self._recurring = _RecurringLapController()
            self._state = _RaceState.RECURRING
            return self._recurring(sensors)
        return self._launch(sensors, progress)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
