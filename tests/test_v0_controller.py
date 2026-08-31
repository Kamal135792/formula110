from __future__ import annotations

from math import isfinite
from pathlib import Path

from controllers.v0 import (
    FEATURE_COUNT,
    Controller,
    TrainingStep,
    default_policy_parameters,
    improve_policy,
    load_policy_parameters,
    save_policy_parameters,
)
from racing import CameraSensors, OdometrySensors, RobotSensors


def test_v0_controller_returns_bounded_finite_actions() -> None:
    controller = Controller(default_policy_parameters())
    sensors = RobotSensors(
        tick=0,
        camera=CameraSensors(
            center_offset_m=0.4,
            heading_error_degrees=8.0,
            lookahead_offsets_m=(0.8, 2.0, 3.5),
        ),
        odometry=OdometrySensors(speed_mps=6.0),
    )

    command = controller(sensors)

    assert isfinite(command.throttle)
    assert isfinite(command.steer)
    assert -1.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0


def test_v0_corner_braking_cannot_latch_until_a_full_stop() -> None:
    controller = Controller(default_policy_parameters())

    def corner_sensors(tick: int) -> RobotSensors:
        return RobotSensors(
            tick=tick,
            camera=CameraSensors(
                center_offset_m=0.0,
                heading_error_degrees=30.0,
                lookahead_offsets_m=(2.0, 6.0, 11.0),
            ),
            odometry=OdometrySensors(speed_mps=32.0),
        )

    brake = controller(corner_sensors(0))
    clear_reverse_request = controller(corner_sensors(3))

    assert brake.throttle < 0.0
    assert clear_reverse_request.throttle == 0.0


def test_v0_policy_gradient_changes_actor_weights() -> None:
    parameters = default_policy_parameters()
    positive_features = (1.0, *([0.25] * (FEATURE_COUNT - 1)))
    negative_features = (1.0, *([-0.25] * (FEATURE_COUNT - 1)))
    episode = [
        TrainingStep(positive_features, (0.08, -0.07), 2.0),
        TrainingStep(negative_features, (-0.08, 0.07), -2.0),
    ]

    updated, metrics = improve_policy(parameters, [episode], discount=1.0)

    assert metrics.decision_count == 2
    assert updated.training_iterations == 1
    assert updated.actor_weights != parameters.actor_weights


def test_v0_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.json"
    parameters = default_policy_parameters()
    parameters.actor_weights[0][0] = 0.125
    parameters.training_iterations = 3
    parameters.best_evaluation_score = 42.5

    save_policy_parameters(parameters, checkpoint)
    loaded = load_policy_parameters(checkpoint)

    assert loaded.actor_weights == parameters.actor_weights
    assert loaded.critic_weights == parameters.critic_weights
    assert loaded.training_iterations == 3
    assert loaded.best_evaluation_score == 42.5
