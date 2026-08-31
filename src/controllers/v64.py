"""V64: first-pass line guard for the wall-sensitive 48-53 m partition."""

from __future__ import annotations

from controllers.track_localizer import TrackLocalizer
from controllers.v63 import Controller as PartitionedController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V64 Damage-Free Handoff Racer"
RACING_COLOR = "#22c55e"


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


class Controller:
    def __init__(
        self,
        *,
        guard_start_m: float = 38.0,
        guard_peak_start_m: float = 44.0,
        guard_peak_end_m: float = 48.0,
        guard_end_m: float = 48.0,
        avoidance_steer: float = -0.175,
        repeat_guard: bool = False,
    ) -> None:
        self._base = PartitionedController()
        self._localizer = TrackLocalizer()
        self._guard_start_m = guard_start_m
        self._guard_peak_start_m = guard_peak_start_m
        self._guard_peak_end_m = guard_peak_end_m
        self._guard_end_m = guard_end_m
        self._avoidance_steer = avoidance_steer
        self._repeat_guard = repeat_guard
        self._guard_active = True
        self._started_beyond_guard = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        if sensors.tick == 0:
            self._guard_active = True
            self._started_beyond_guard = progress >= self._guard_end_m

        command = self._base(sensors)
        if not self._guard_active:
            return command

        if self._started_beyond_guard:
            if progress < self._guard_start_m:
                self._started_beyond_guard = False
            else:
                return command

        if progress >= self._guard_end_m:
            if not self._repeat_guard:
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
        steer = _clamp(command.steer + self._avoidance_steer * _clamp(weight, 0.0, 1.0), -1.0, 1.0)
        return RobotCommand(throttle=command.throttle, steer=steer)

    def copy_for_car(self) -> Controller:
        return Controller(
            guard_start_m=self._guard_start_m,
            guard_peak_start_m=self._guard_peak_start_m,
            guard_peak_end_m=self._guard_peak_end_m,
            guard_end_m=self._guard_end_m,
            avoidance_steer=self._avoidance_steer,
            repeat_guard=self._repeat_guard,
        )


def create_controller() -> Controller:
    return Controller()
