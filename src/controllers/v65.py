"""V65: predictive right-wall safety shield for the first corner pass."""

from __future__ import annotations

from controllers.track_localizer import TrackLocalizer
from controllers.v63 import Controller as PartitionedController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V65 Predictive Zero-Damage Racer"
RACING_COLOR = "#10b981"


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


class Controller:
    def __init__(
        self,
        *,
        clearance_trigger_m: float = 3.0,
        full_response_clearance_m: float = 1.0,
        full_response_steer: float = -0.85,
        throttle_cap: float = 0.0,
    ) -> None:
        self._base = PartitionedController()
        self._localizer = TrackLocalizer()
        self._clearance_trigger_m = clearance_trigger_m
        self._full_response_clearance_m = full_response_clearance_m
        self._full_response_steer = full_response_steer
        self._throttle_cap = throttle_cap
        self._shield_active = True
        self._started_beyond_partition = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        if sensors.tick == 0:
            self._shield_active = True
            self._started_beyond_partition = progress >= 56.0
        command = self._base(sensors)
        if not self._shield_active:
            return command
        if self._started_beyond_partition:
            if progress < 42.0:
                self._started_beyond_partition = False
            else:
                return command
        if progress >= 56.0:
            self._shield_active = False
            return command
        if progress < 42.0 or sensors.wall_lidar.right_m >= self._clearance_trigger_m:
            return command

        urgency = _clamp(
            (self._clearance_trigger_m - sensors.wall_lidar.right_m)
            / (self._clearance_trigger_m - self._full_response_clearance_m),
            0.0,
            1.0,
        )
        shield_steer = -0.35 + urgency * (self._full_response_steer + 0.35)
        return RobotCommand(
            throttle=min(command.throttle, self._throttle_cap),
            steer=min(command.steer, shield_steer),
        )

    def copy_for_car(self) -> Controller:
        return Controller(
            clearance_trigger_m=self._clearance_trigger_m,
            full_response_clearance_m=self._full_response_clearance_m,
            full_response_steer=self._full_response_steer,
            throttle_cap=self._throttle_cap,
        )


def create_controller() -> Controller:
    return Controller()
