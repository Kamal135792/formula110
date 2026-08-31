"""V4: deterministic fast geometric planner without the learned residual."""

from __future__ import annotations

from controllers.v0 import Controller as GeometricController
from controllers.v0 import default_policy_parameters

RACING_NAME = "V4 Geometric Racer"
RACING_COLOR = "#06b6d4"


class Controller(GeometricController):
    """Use V0's planner while deliberately removing checkpoint dependence."""

    def __init__(self) -> None:
        super().__init__(default_policy_parameters())

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
