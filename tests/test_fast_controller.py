from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from math import inf
from pathlib import Path

import pytest

from controllers.fast_controller import Controller as FastController
from controllers.fast_controller import create_controller
from controllers.v75 import Controller as V75Controller
from racing import (
    CameraSensors,
    ImuSensors,
    LidarSensors,
    OdometrySensors,
    RobotCommand,
    RobotSensors,
    load_student_submission,
)

ROOT = Path(__file__).parents[1]
MAP_PATH = ROOT / "src" / "controllers" / "track_signature_map.json"
MAP_SAMPLES = json.loads(MAP_PATH.read_text(encoding="utf-8"))["samples"]


def _sensor_sequence(
    start_m: float,
    ticks: int,
    *,
    front_m: float = inf,
    classifier_heading_degrees: float | None = None,
) -> Iterator[RobotSensors]:
    start_index = min(range(len(MAP_SAMPLES)), key=lambda index: abs(MAP_SAMPLES[index][0] - start_m))
    lidar = LidarSensors(distances_m=(20.0, 20.0, 20.0, front_m, 20.0, 20.0, 20.0))
    for tick in range(ticks):
        _progress, desired_heading, near, middle, far = MAP_SAMPLES[(start_index + tick) % len(MAP_SAMPLES)]
        heading = desired_heading if classifier_heading_degrees is None else classifier_heading_degrees
        yield RobotSensors(
            dt_s=1.0 / 60.0,
            tick=tick,
            imu=ImuSensors(heading_degrees=heading),
            odometry=OdometrySensors(speed_mps=20.0, distance_m=0.5 * tick),
            lidar=lidar,
            wall_lidar=lidar,
            camera=CameraSensors(
                heading_error_degrees=desired_heading - heading,
                lookahead_offsets_m=(near, middle, far),
            ),
        )


@pytest.mark.parametrize(
    ("start_m", "ticks", "front_m", "heading"),
    [
        (0.0, 400, inf, None),
        (10.5, 160, 38.0, None),
        (24.0, 160, 43.0, None),
        (40.0, 160, inf, 45.0),
        (40.0, 160, inf, 60.0),
        (179.0, 280, inf, None),
    ],
)
def test_fast_controller_matches_v75_on_identical_public_sensor_sequences(
    start_m: float,
    ticks: int,
    front_m: float,
    heading: float | None,
) -> None:
    baseline = V75Controller()
    fast_controller = FastController()

    for sensors in _sensor_sequence(
        start_m,
        ticks,
        front_m=front_m,
        classifier_heading_degrees=heading,
    ):
        assert fast_controller(sensors) == baseline(sensors), f"command mismatch at tick {sensors.tick}"


def test_fast_controller_factory_returns_independent_runtime_state() -> None:
    first = create_controller()
    second = create_controller()

    assert first is not second
    for sensors in _sensor_sequence(0.0, 30):
        assert first(sensors) == second(sensors)


def test_fast_controller_loads_through_the_public_submission_api() -> None:
    submission = load_student_submission("controllers.fast_controller")

    command = submission.controller(next(_sensor_sequence(0.0, 1)))

    assert isinstance(command, RobotCommand)
    assert submission.display_name == "Fast Hybrid Racer"
    assert submission.car_color == (0x3F / 255, 0x62 / 255, 0x12 / 255, 1.0)


def test_fast_controller_has_no_historical_controller_imports_or_training_structures() -> None:
    source_path = ROOT / "src" / "controllers" / "fast_controller.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("controllers.v") for module in imported_modules)
    assert "TrainingStep" not in source
    assert "LearningMetrics" not in source
    assert "critic" not in source.lower()
    assert "random" not in source.lower()
    assert "reward" not in source.lower()
    assert "checkpoint" not in source.lower()
