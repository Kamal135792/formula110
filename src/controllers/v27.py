"""V27: 60 Hz control and fine brake pulses at the 36.3 m/s envelope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from controllers.v0 import Controller as V0Controller
from controllers.v0 import PlannerConfig, load_policy_parameters

RACING_NAME = "V27 60Hz Precision Racer"
RACING_COLOR = "#f472b6"
_CONFIG_PATH = Path(__file__).with_name("v25_config.json")
_WEIGHTS_PATH = Path(__file__).with_name("v21_weights.json")


def _config() -> PlannerConfig:
    record = cast(dict[str, object], json.loads(_CONFIG_PATH.read_text(encoding="utf-8")))
    return PlannerConfig(**cast(dict[str, float], record["planner"]))


class Controller(V0Controller):
    def __init__(self) -> None:
        super().__init__(
            load_policy_parameters(_WEIGHTS_PATH),
            planner_config=_config(),
            decision_interval_ticks=1,
        )

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
