"""V40: spawn-safe controller with a localized start/finish straight attack."""

from __future__ import annotations

from math import isfinite

from controllers.track_localizer import TrackLocalizer
from controllers.v24 import Controller as StableController
from controllers.v38 import Controller as LaunchApexController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V40 Partitioned Straight Racer"
RACING_COLOR = "#22d3ee"

BOOST_START_M = 165.0
BOOST_END_M = 2.0
BOOST_TARGET_SPEED_MPS = 36.5


def _in_wrapped_partition(progress_m: float, start_m: float, end_m: float) -> bool:
    if start_m <= end_m:
        return start_m <= progress_m < end_m
    return progress_m >= start_m or progress_m < end_m


class Controller:
    def __init__(
        self,
        *,
        boost_start_m: float = BOOST_START_M,
        boost_end_m: float = BOOST_END_M,
        boost_target_speed_mps: float = BOOST_TARGET_SPEED_MPS,
    ) -> None:
        self._localizer = TrackLocalizer()
        self._stable = StableController()
        self._launch_apex = LaunchApexController()
        self._boost_start_m = boost_start_m
        self._boost_end_m = boost_end_m
        self._boost_target_speed_mps = boost_target_speed_mps
        self._use_stable = False

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        progress = self._localizer.update(sensors)
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
            self._use_stable = straight_hazard or curve_hazard

        stable = self._stable(sensors)
        launch_apex = self._launch_apex(sensors)
        if self._use_stable:
            return stable

        # Keep the proven one-second launch untouched.  On later laps, attack
        # the entire long straight and hand control back well before the bend.
        if (
            sensors.tick >= 60
            and _in_wrapped_partition(progress, self._boost_start_m, self._boost_end_m)
            and launch_apex.throttle >= 0.0
            and sensors.odometry.speed_mps < self._boost_target_speed_mps
        ):
            return RobotCommand(throttle=1.0, steer=launch_apex.steer)
        return launch_apex

    def copy_for_car(self) -> Controller:
        return Controller(
            boost_start_m=self._boost_start_m,
            boost_end_m=self._boost_end_m,
            boost_target_speed_mps=self._boost_target_speed_mps,
        )


def create_controller() -> Controller:
    return Controller()
