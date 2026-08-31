"""Imitation-refine a V0 policy from recorded human keyboard/gamepad laps."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import json
from math import tanh
from pathlib import Path
from typing import Any, cast

from controllers.v0 import (
    _ACTION_RESIDUAL_SCALES,
    ACTION_COUNT,
    FEATURE_COUNT,
    Controller,
    PlannerConfig,
    _policy_features,
    default_policy_parameters,
    load_policy_parameters,
    save_policy_parameters,
)
from racing import (
    CameraSensors,
    ContactSensors,
    ImuSensors,
    LidarSensors,
    OdometrySensors,
    RobotCommand,
    RobotSensors,
)


def _float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _mapping(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value)


def _lidar(value: Any) -> LidarSensors:
    record = _mapping(value)
    distances = tuple(_float(item, float("inf")) for item in cast(list[Any], record["distances_m"]))
    return LidarSensors(
        angles_degrees=tuple(float(item) for item in cast(list[Any], record["angles_degrees"])),
        distances_m=distances,
        max_distance_m=_float(record.get("max_distance_m"), float("inf")),
    )


def _sensors(value: Any) -> RobotSensors:
    record = _mapping(value)
    imu = _mapping(record["imu"])
    odometry = _mapping(record["odometry"])
    camera = _mapping(record["camera"])
    contact = _mapping(record["contact"])
    return RobotSensors(
        dt_s=float(record["dt_s"]),
        tick=int(record["tick"]),
        imu=ImuSensors(**cast(dict[str, float], imu)),
        odometry=OdometrySensors(**cast(dict[str, float], odometry)),
        lidar=_lidar(record["lidar"]),
        wall_lidar=_lidar(record["wall_lidar"]),
        camera=CameraSensors(
            visible=bool(camera["visible"]),
            center_offset_m=float(camera["center_offset_m"]),
            heading_error_degrees=float(camera["heading_error_degrees"]),
            lookahead_offsets_m=tuple(float(item) for item in cast(list[Any], camera["lookahead_offsets_m"])),
            lookahead_distances_m=tuple(float(item) for item in cast(list[Any], camera["lookahead_distances_m"])),
        ),
        contact=ContactSensors(**cast(dict[str, float], contact)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="JSONL produced with --record-human")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("src/controllers/v8_weights.json"))
    parser.add_argument("--output", type=Path, default=Path("src/controllers/v11_weights.json"))
    args = parser.parse_args()

    examples: list[tuple[RobotSensors, RobotCommand]] = []
    for line in args.recording.read_text(encoding="utf-8").splitlines():
        record = _mapping(json.loads(line))
        if record.get("record_type") != "human_control_step":
            continue
        command = _mapping(record["command"])
        examples.append(
            (_sensors(record["sensors"]), RobotCommand(float(command["throttle"]), float(command["steer"])))
        )
    if not examples:
        raise ValueError("recording contains no human_control_step records")

    parameters = load_policy_parameters(args.initial_checkpoint)
    planner_record = _mapping(json.loads(Path("src/controllers/v7_config.json").read_text(encoding="utf-8")))
    planner = PlannerConfig(**cast(dict[str, Any], planner_record["planner"]))
    for _epoch in range(args.epochs):
        baseline = Controller(default_policy_parameters(), planner_config=planner)
        loss = 0.0
        for sensors, human in examples:
            base = baseline(sensors)
            features = _policy_features(sensors, base, base)
            for action, target, scale in (
                (0, human.throttle, _ACTION_RESIDUAL_SCALES[0]),
                (1, human.steer, _ACTION_RESIDUAL_SCALES[1]),
            ):
                activation = sum(
                    weight * feature for weight, feature in zip(parameters.actor_weights[action], features, strict=True)
                )
                tanh_activation = tanh(activation)
                prediction = (
                    base.throttle + scale * tanh_activation if action == 0 else base.steer + scale * tanh_activation
                )
                error = max(-1.0, min(1.0, prediction - target))
                loss += error * error
                derivative = 2.0 * error * scale * (1.0 - tanh_activation * tanh_activation)
                for index in range(FEATURE_COUNT):
                    parameters.actor_weights[action][index] -= args.learning_rate * derivative * features[index]
        print(f"epoch {_epoch + 1}: mean_action_mse={loss / (len(examples) * ACTION_COUNT):.6f}")
    save_policy_parameters(parameters, args.output)
    print(f"saved imitation-refined policy from {len(examples)} human frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
