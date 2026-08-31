"""V69: safe spawn expert followed by a fresh universal recurring-lap racer."""

from __future__ import annotations

from controllers.track_localizer import TrackLocalizer
from controllers.v41 import Controller as RecurringLapController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS
from controllers.v66 import Controller as LaunchPathMixtureController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V69 Launch-and-Recurring Two-Stage Racer"
RACING_COLOR = "#064e3b"


class Controller:
    def __init__(
        self,
        *,
        recurring_corner_offsets_m: tuple[float, ...] = LEARNED_CORNER_OFFSETS_M,
        recurring_curve_speed_gains: tuple[float, ...] = ROBUST_LIMIT_CURVE_SPEED_GAINS,
        recurring_speed_boundaries_m: tuple[float, ...] = AGGRESSIVE_BRAKING_BOUNDARIES_M,
        recurring_straight_speed_mps: float = AGGRESSIVE_STRAIGHT_SPEED_MPS,
    ) -> None:
        self._launch = LaunchPathMixtureController()
        self._race: RecurringLapController | None = None
        self._localizer = TrackLocalizer()
        self._started_beyond_merge = False
        self._recurring_corner_offsets_m = recurring_corner_offsets_m
        self._recurring_curve_speed_gains = recurring_curve_speed_gains
        self._recurring_speed_boundaries_m = recurring_speed_boundaries_m
        self._recurring_straight_speed_mps = recurring_straight_speed_mps

    def _new_race_controller(self) -> RecurringLapController:
        return RecurringLapController(
            corner_offsets_m=self._recurring_corner_offsets_m,
            curve_speed_gains=self._recurring_curve_speed_gains,
            speed_boundaries_m=self._recurring_speed_boundaries_m,
            straight_speed_numerator=self._recurring_straight_speed_mps,
        )

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        if sensors.tick == 0:
            self._started_beyond_merge = progress >= 60.0
        if self._race is not None:
            return self._race(sensors)
        if self._started_beyond_merge:
            if progress < 10.0:
                self._started_beyond_merge = False
        elif progress >= 60.0:
            self._race = self._new_race_controller()
            return self._race(sensors)
        return self._launch(sensors)

    def copy_for_car(self) -> Controller:
        return Controller(
            recurring_corner_offsets_m=self._recurring_corner_offsets_m,
            recurring_curve_speed_gains=self._recurring_curve_speed_gains,
            recurring_speed_boundaries_m=self._recurring_speed_boundaries_m,
            recurring_straight_speed_mps=self._recurring_straight_speed_mps,
        )


def create_controller() -> Controller:
    return Controller()
