"""V29: phase-aware outside-inside racing line layered over V24."""

from __future__ import annotations

from controllers.v24 import Controller as BaseController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V29 Racing Line Racer"
RACING_COLOR = "#38bdf8"
OUTSIDE_OFFSET_M = 0.0
INSIDE_OFFSET_M = 0.60
LINE_STEER_GAIN = 0.06


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


class Controller:
    def __init__(self) -> None:
        self._base = BaseController()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        command = self._base(sensors)
        if sensors.contact.wall > 0.0 or not sensors.camera.visible:
            return command

        offsets = (*sensors.camera.lookahead_offsets_m, 0.0, 0.0, 0.0)
        near_curve = _clamp(abs(offsets[0]) / 2.2, 0.0, 1.0)
        far_curve = _clamp(abs(offsets[2]) / 8.0, 0.0, 1.0)
        approach = _clamp((far_curve - near_curve) * 2.8, 0.0, 1.0)
        apex = _clamp(near_curve * 1.8, 0.0, 1.0)
        direction_source = offsets[2] if abs(offsets[2]) > abs(offsets[0]) else offsets[0]
        direction = 1.0 if direction_source > 0.0 else -1.0 if direction_source < 0.0 else 0.0

        desired_car_offset = direction * (INSIDE_OFFSET_M * apex - OUTSIDE_OFFSET_M * approach)
        correction = _clamp(LINE_STEER_GAIN * desired_car_offset, -0.08, 0.08)
        return RobotCommand(throttle=command.throttle, steer=_clamp(command.steer + correction, -1.0, 1.0))

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
