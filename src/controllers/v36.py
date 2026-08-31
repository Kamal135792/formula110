"""V36: track-localized launch, apex, and straight partition controller."""

from __future__ import annotations

from controllers.track_localizer import TrackLocalizer
from controllers.v24 import Controller as StableController
from controllers.v31 import Controller as ApexController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V36 Partition Racer"
RACING_COLOR = "#00b8d4"
STABLE_LAUNCH_TICKS = 300
BOOST_START_M = 164.0
BOOST_END_M = 174.0
BOOST_ENABLED = True


class Controller:
    def __init__(self, *, boost_enabled: bool = BOOST_ENABLED, stable_launch_ticks: int = STABLE_LAUNCH_TICKS) -> None:
        self._localizer = TrackLocalizer()
        self._stable = StableController()
        self._apex = ApexController()
        self._boost_enabled = boost_enabled
        self._stable_launch_ticks = stable_launch_ticks

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
        stable = self._stable(sensors)
        apex = self._apex(sensors)
        if sensors.tick < self._stable_launch_ticks:
            return stable

        throttle = apex.throttle
        in_boost_partition = BOOST_START_M <= progress < BOOST_END_M
        if self._boost_enabled and in_boost_partition and apex.throttle > 0.0 and sensors.odometry.speed_mps < 36.5:
            throttle = 1.0
        return RobotCommand(throttle=throttle, steer=apex.steer)

    def copy_for_car(self) -> Controller:
        return Controller(boost_enabled=self._boost_enabled, stable_launch_ticks=self._stable_launch_ticks)


def create_controller() -> Controller:
    return Controller()
