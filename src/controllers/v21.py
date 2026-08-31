"""V21: 36 m/s policy retrained and selected across ten random starts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from controllers.v0 import Controller as V0Controller
from controllers.v0 import PlannerConfig, load_policy_parameters

RACING_NAME = "V21 Retrained 36mps Racer"
RACING_COLOR = "#34d399"
_CONFIG_PATH = Path(__file__).with_name("v12_config.json")
_WEIGHTS_PATH = Path(__file__).with_name("v21_weights.json")


def _config() -> PlannerConfig:
    record = cast(dict[str, object], json.loads(_CONFIG_PATH.read_text(encoding="utf-8")))
    return PlannerConfig(**cast(dict[str, float], record["planner"]))


class Controller(V0Controller):
    def __init__(self) -> None:
        super().__init__(load_policy_parameters(_WEIGHTS_PATH), planner_config=_config())

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
