"""V72: V71 with separate smooth apex and exit controls in the final corner."""

from __future__ import annotations

from controllers.track_localizer import TrackLocalizer
from controllers.v41 import Controller as SectorController
from controllers.v69 import Controller as TwoStageController
from controllers.v70 import RECOMBINED_INSIDE_OFFSETS_M
from controllers.v71 import SECTOR_2_LIMIT_GAINS
from racing import RobotCommand, RobotSensors

RACING_NAME = "V72 Final-Corner Full-Width Racer"
RACING_COLOR = "#84cc16"


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _triangle(progress_m: float, start_m: float, peak_m: float, end_m: float) -> float:
    if progress_m <= start_m or progress_m >= end_m:
        return 0.0
    if progress_m <= peak_m:
        return (progress_m - start_m) / (peak_m - start_m)
    return (end_m - progress_m) / (end_m - peak_m)


class FinalCornerController(SectorController):
    def __init__(
        self,
        *,
        apex_steer: float,
        exit_steer: float,
        exit_start_m: float,
        exit_peak_m: float,
        exit_end_m: float,
        exit_throttle_floor: float | None,
        exit_throttle_start_m: float,
        exit_throttle_end_m: float,
        final_curve_speed_gain: float,
        straight_speed_mps: float,
    ) -> None:
        gains = (*SECTOR_2_LIMIT_GAINS[:4], final_curve_speed_gain)
        super().__init__(
            corner_offsets_m=RECOMBINED_INSIDE_OFFSETS_M,
            curve_speed_gains=gains,
            speed_boundaries_m=(8.0, 48.0, 82.0, 115.0, 140.0, 165.0),
            straight_speed_numerator=straight_speed_mps,
        )
        self._profile_localizer = TrackLocalizer()
        self._apex_steer = apex_steer
        self._exit_steer = exit_steer
        self._exit_start_m = exit_start_m
        self._exit_peak_m = exit_peak_m
        self._exit_end_m = exit_end_m
        self._exit_throttle_floor = exit_throttle_floor
        self._exit_throttle_start_m = exit_throttle_start_m
        self._exit_throttle_end_m = exit_throttle_end_m

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._profile_localizer.update(sensors)
        command = super().__call__(sensors)
        correction = self._apex_steer * _triangle(progress, 132.0, 147.0, 155.0)
        correction += self._exit_steer * _triangle(
            progress,
            self._exit_start_m,
            self._exit_peak_m,
            self._exit_end_m,
        )
        throttle = command.throttle
        if (
            self._exit_throttle_floor is not None
            and self._exit_throttle_start_m <= progress <= self._exit_throttle_end_m
        ):
            throttle = max(throttle, self._exit_throttle_floor)
        return RobotCommand(
            throttle=throttle,
            steer=_clamp(command.steer + correction, -1.0, 1.0),
        )


class Controller(TwoStageController):
    def __init__(
        self,
        *,
        apex_steer: float = 0.04,
        exit_steer: float = -0.09,
        final_curve_speed_gain: float = 0.78,
        straight_speed_mps: float = 37.5,
        exit_start_m: float = 148.0,
        exit_peak_m: float = 160.0,
        exit_end_m: float = 170.0,
        exit_throttle_floor: float | None = None,
        exit_throttle_start_m: float = 152.0,
        exit_throttle_end_m: float = 165.0,
    ) -> None:
        super().__init__(
            recurring_corner_offsets_m=RECOMBINED_INSIDE_OFFSETS_M,
            recurring_curve_speed_gains=SECTOR_2_LIMIT_GAINS,
        )
        self._apex_steer = apex_steer
        self._exit_steer = exit_steer
        self._final_curve_speed_gain = final_curve_speed_gain
        self._straight_speed_mps = straight_speed_mps
        self._exit_start_m = exit_start_m
        self._exit_peak_m = exit_peak_m
        self._exit_end_m = exit_end_m
        self._exit_throttle_floor = exit_throttle_floor
        self._exit_throttle_start_m = exit_throttle_start_m
        self._exit_throttle_end_m = exit_throttle_end_m

    def _new_race_controller(self) -> SectorController:
        return FinalCornerController(
            apex_steer=self._apex_steer,
            exit_steer=self._exit_steer,
            exit_start_m=self._exit_start_m,
            exit_peak_m=self._exit_peak_m,
            exit_end_m=self._exit_end_m,
            exit_throttle_floor=self._exit_throttle_floor,
            exit_throttle_start_m=self._exit_throttle_start_m,
            exit_throttle_end_m=self._exit_throttle_end_m,
            final_curve_speed_gain=self._final_curve_speed_gain,
            straight_speed_mps=self._straight_speed_mps,
        )

    def copy_for_car(self) -> Controller:
        return Controller(
            apex_steer=self._apex_steer,
            exit_steer=self._exit_steer,
            final_curve_speed_gain=self._final_curve_speed_gain,
            straight_speed_mps=self._straight_speed_mps,
            exit_start_m=self._exit_start_m,
            exit_peak_m=self._exit_peak_m,
            exit_end_m=self._exit_end_m,
            exit_throttle_floor=self._exit_throttle_floor,
            exit_throttle_start_m=self._exit_throttle_start_m,
            exit_throttle_end_m=self._exit_throttle_end_m,
        )


def create_controller() -> Controller:
    return Controller()
