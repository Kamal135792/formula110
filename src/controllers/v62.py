"""V62: V61 plus a 90-tick launch for the deepest hairpin starts."""

from __future__ import annotations

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS
from controllers.v60 import Controller as SplitHairpinController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V62 Dual-Hairpin Launch Racer"
RACING_COLOR = "#ec4899"
DEEP_HAIRPIN_STABILIZATION_TICKS = 90


class Controller(SplitHairpinController):
    def __init__(self) -> None:
        super().__init__(extended_curve_min_heading=50.0)
        self._deep_hairpin = SectorController(
            corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
            curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
            speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
            straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
            hazard_stable_ticks=DEEP_HAIRPIN_STABILIZATION_TICKS,
            extended_start_hazards=True,
        )

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            heading = sensors.imu.heading_degrees
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            if 40.0 < heading < 50.0 and far_offset < -7.0:
                self._selected = self._deep_hairpin
                return self._selected(sensors)
        return super().__call__(sensors)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
