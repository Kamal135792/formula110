"""V5: multi-seed optimized scaling of the learned V0 policy."""

from __future__ import annotations

from pathlib import Path

from controllers.v0 import Controller as V0Controller
from controllers.v0 import load_policy_parameters

RACING_NAME = "V5 Tuned Policy Racer"
RACING_COLOR = "#f43f5e"
_WEIGHTS = Path(__file__).with_name("v5_weights.json")


class Controller(V0Controller):
    def __init__(self) -> None:
        super().__init__(load_policy_parameters(_WEIGHTS))

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
