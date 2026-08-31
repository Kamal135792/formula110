"""V1: learned V0 policy with predictive wall-clearance supervision."""

from __future__ import annotations

from math import isfinite

from controllers.v0 import Controller as V0Controller
from racing import RobotCommand, RobotSensors

RACING_NAME = "V1 Predictive Racer"
RACING_COLOR = "#ff8a2d"


def _finite_distance(value: float, fallback: float = 30.0) -> float:
    return value if isfinite(value) else fallback


class Controller:
    """Add anticipatory wall clearance without replacing V0's learned policy."""

    def __init__(self) -> None:
        self._base = V0Controller()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        command = self._base(sensors)
        # Seeded spawns can begin midway through a bend. Give the base policy
        # two seconds to settle before adding a clearance bias; raw side ranges
        # are ambiguous until the car is aligned with the local track tangent.
        if sensors.tick < 120 or sensors.contact.wall > 0.0 or not sensors.camera.visible:
            return command

        left = _finite_distance(sensors.wall_lidar.front_left_m)
        right = _finite_distance(sensors.wall_lidar.front_right_m)
        front = _finite_distance(sensors.wall_lidar.front_m)
        speed = max(0.0, sensors.odometry.speed_mps)

        # Side clearance becomes important farther ahead as speed rises. Bias
        # smoothly toward open pavement instead of waiting for contact.
        clearance_error = max(-1.0, min(1.0, (left - right) / 7.0))
        proximity = max(0.0, min(1.0, (5.5 + 0.10 * speed - min(left, right)) / 4.5))
        curve_direction = sensors.camera.heading_error_degrees
        line_agreement = max(0.0, min(1.0, abs(curve_direction) / 20.0))
        # Never allow clearance supervision to reverse the intended turn.
        correction = -0.20 * proximity * clearance_error * line_agreement
        if command.steer * correction < 0.0:
            correction = max(-0.35 * abs(command.steer), min(0.35 * abs(command.steer), correction))
        steer = command.steer + correction

        # Only cap acceleration for a wall directly in the predicted path.
        # Coasting preserves momentum and avoids V0's reverse/brake latch.
        stopping_margin = front - (2.0 + 0.12 * speed)
        throttle = command.throttle
        if stopping_margin < 0.0 and abs(command.steer) > 0.18:
            throttle = min(throttle, 0.0)
        return RobotCommand(
            throttle=max(-1.0, min(1.0, throttle)),
            steer=max(-1.0, min(1.0, steer)),
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
