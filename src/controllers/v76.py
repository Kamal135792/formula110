"""V76: 60 Hz periodic-spline follower seeded from V75's recurring line."""

from __future__ import annotations

from math import floor

from controllers.track_localizer import TrackLocalizer
from controllers.v69 import Controller as TwoStageController
from controllers.v72 import FinalCornerController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V76 Periodic Spline Trajectory Racer"
RACING_COLOR = "#0891b2"

SPLINE_CONTROL_OFFSETS_M = (
    0.009,
    0.197,
    0.534,
    1.510,
    -0.554,
    -2.991,
    -0.001,
    1.411,
    -1.411,
    -2.210,
    -0.512,
    1.373,
    0.689,
    1.251,
    0.088,
    0.486,
    -0.047,
    -0.283,
)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _periodic_catmull_rom(values: tuple[float, ...], progress_m: float, total_length_m: float) -> float:
    position = ((progress_m - 5.0) % total_length_m) * len(values) / total_length_m
    index = floor(position)
    t = position - index
    p0 = values[(index - 1) % len(values)]
    p1 = values[index % len(values)]
    p2 = values[(index + 1) % len(values)]
    p3 = values[(index + 2) % len(values)]
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


class SplineRecurringController(FinalCornerController):
    def __init__(self, *, control_offsets_m: tuple[float, ...], follower_gain: float) -> None:
        if len(control_offsets_m) != len(SPLINE_CONTROL_OFFSETS_M):
            raise ValueError(f"control_offsets_m must contain {len(SPLINE_CONTROL_OFFSETS_M)} values")
        super().__init__(
            apex_steer=0.04,
            exit_steer=-0.09,
            exit_start_m=148.0,
            exit_peak_m=160.0,
            exit_end_m=170.0,
            exit_throttle_floor=1.0,
            exit_throttle_start_m=157.0,
            exit_throttle_end_m=165.0,
            final_curve_speed_gain=0.78,
            straight_speed_mps=39.0,
        )
        self._spline_localizer = TrackLocalizer()
        self._control_offsets_m = control_offsets_m
        self._follower_gain = follower_gain

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._spline_localizer.update(sensors)
        command = super().__call__(sensors)
        if not sensors.camera.visible or self._follower_gain == 0.0:
            return command
        desired_offset = _periodic_catmull_rom(
            self._control_offsets_m,
            progress,
            self._spline_localizer.total_length_m,
        )
        lateral_error = sensors.camera.center_offset_m - desired_offset
        return RobotCommand(
            throttle=command.throttle,
            steer=_clamp(command.steer + self._follower_gain * lateral_error, -1.0, 1.0),
        )


class Controller(TwoStageController):
    def __init__(
        self,
        *,
        control_offsets_m: tuple[float, ...] = SPLINE_CONTROL_OFFSETS_M,
        follower_gain: float = 0.04,
    ) -> None:
        super().__init__()
        self._control_offsets_m = control_offsets_m
        self._follower_gain = follower_gain

    def _new_race_controller(self) -> SplineRecurringController:
        return SplineRecurringController(
            control_offsets_m=self._control_offsets_m,
            follower_gain=self._follower_gain,
        )

    def copy_for_car(self) -> Controller:
        return Controller(
            control_offsets_m=self._control_offsets_m,
            follower_gain=self._follower_gain,
        )


def create_controller() -> Controller:
    return Controller()
