"""V7: faster multi-seed optimized geometric planner plus learned policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from controllers.v0 import Controller as V0Controller
from controllers.v0 import PlannerConfig

RACING_NAME = "V7 Fast Robust Racer"
RACING_COLOR = "#14b8a6"
_CONFIG_PATH = Path(__file__).with_name("v7_config.json")


def _planner_config() -> PlannerConfig:
    record = cast(dict[str, object], json.loads(_CONFIG_PATH.read_text(encoding="utf-8")))
    values = cast(dict[str, float], record["planner"])
    return PlannerConfig(**values)


class Controller(V0Controller):
    def __init__(self) -> None:
        super().__init__(planner_config=_planner_config())

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
