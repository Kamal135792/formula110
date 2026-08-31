"""V61: V60 with the hairpin launch split at its measured signature boundary."""

from __future__ import annotations

from controllers.v60 import Controller as AuditedController

RACING_NAME = "V61 Split-Hairpin Spawn Racer"
RACING_COLOR = "#d946ef"
HAIRPIN_STAGING_MIN_HEADING_DEGREES = 50.0


class Controller(AuditedController):
    def __init__(self) -> None:
        super().__init__(extended_curve_min_heading=HAIRPIN_STAGING_MIN_HEADING_DEGREES)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
