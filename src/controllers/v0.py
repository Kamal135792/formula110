"""Fast track-following controller with a lightweight learned residual policy.

The deterministic part supplies safe behaviour before training.  The training
script in ``scripts/train_v0.py`` improves a small Gaussian policy with an
episodic policy-gradient update and stores its best weights next to this file.
Inference uses only the learned mean action and has no random exploration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import atan, atan2, cos, inf, isfinite, radians, sqrt, tanh
from pathlib import Path
from random import Random
from typing import cast

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "V0 RL Racer"
RACING_COLOR: str = "#2d8cff"

FEATURE_COUNT = 15
ACTION_COUNT = 2
POLICY_SCHEMA_VERSION = 1
DECISION_INTERVAL_TICKS = 3

_ACTION_THROTTLE = 0
_ACTION_STEER = 1
_ACTION_RESIDUAL_SCALES = (0.30, 0.25)
_EXPLORATION_STANDARD_DEVIATIONS = (0.08, 0.07)
_CHECKPOINT_PATH = Path(__file__).with_name("v0_weights.json")
_WHEELBASE_M = 1.40
_MAX_STEERING_RADIANS = radians(25.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(slots=True)
class PolicyParameters:
    """Trainable actor and critic parameters small enough for pure Python."""

    actor_weights: list[list[float]]
    critic_weights: list[float]
    training_iterations: int = 0
    best_evaluation_score: float | None = None

    def copy(self) -> PolicyParameters:
        return PolicyParameters(
            actor_weights=[row.copy() for row in self.actor_weights],
            critic_weights=self.critic_weights.copy(),
            training_iterations=self.training_iterations,
            best_evaluation_score=self.best_evaluation_score,
        )


@dataclass(frozen=True, slots=True)
class TrainingStep:
    """One policy decision and the reward accumulated while it was active."""

    features: tuple[float, ...]
    action_noise: tuple[float, float]
    reward: float


@dataclass(frozen=True, slots=True)
class LearningMetrics:
    """Small diagnostic returned after one policy-gradient update."""

    episode_count: int
    decision_count: int
    mean_discounted_return: float
    actor_gradient_norm: float


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Tunable geometric planner constants; defaults preserve V0 behavior."""

    pure_pursuit_gain: float = 1.65
    heading_gain: float = 0.34
    center_gain: float = 0.12
    speed_numerator: float = 33.0
    curve_speed_gain: float = 1.15
    minimum_corner_speed: float = 12.0
    apex_bias_m: float = 0.0


def default_policy_parameters() -> PolicyParameters:
    """Return the zero-residual policy used when no checkpoint exists."""
    return PolicyParameters(
        actor_weights=[[0.0 for _ in range(FEATURE_COUNT)] for _ in range(ACTION_COUNT)],
        critic_weights=[0.0 for _ in range(FEATURE_COUNT)],
    )


def load_policy_parameters(path: Path = _CHECKPOINT_PATH) -> PolicyParameters:
    """Load a validated JSON policy, or the safe zero policy if absent."""
    if not path.exists():
        return default_policy_parameters()
    unchecked_record: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(unchecked_record, dict):
        raise ValueError("v0 policy checkpoint must contain a JSON object")
    record = cast(dict[str, object], unchecked_record)
    if record.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported v0 policy checkpoint schema")

    actor_weights = _numeric_matrix(record.get("actor_weights"), ACTION_COUNT, FEATURE_COUNT, "actor_weights")
    critic_weights = _numeric_vector(record.get("critic_weights"), FEATURE_COUNT, "critic_weights")
    iterations_value = record.get("training_iterations", 0)
    if not isinstance(iterations_value, int) or isinstance(iterations_value, bool) or iterations_value < 0:
        raise ValueError("training_iterations must be a non-negative integer")
    score_value = record.get("best_evaluation_score")
    best_score = None if score_value is None else _finite_number(score_value, "best_evaluation_score")
    return PolicyParameters(actor_weights, critic_weights, iterations_value, best_score)


def save_policy_parameters(parameters: PolicyParameters, path: Path = _CHECKPOINT_PATH) -> None:
    """Atomically save policy parameters for later controller inference."""
    record: dict[str, object] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "action_count": ACTION_COUNT,
        "actor_weights": parameters.actor_weights,
        "critic_weights": parameters.critic_weights,
        "training_iterations": parameters.training_iterations,
        "best_evaluation_score": parameters.best_evaluation_score,
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _numeric_matrix(value: object, rows: int, columns: int, name: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must contain {rows} rows")
    items = cast(list[object], value)
    if len(items) != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    return [_numeric_vector(row, columns, f"{name}[{index}]") for index, row in enumerate(items)]


def _numeric_vector(value: object, length: int, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must contain {length} values")
    items = cast(list[object], value)
    if len(items) != length:
        raise ValueError(f"{name} must contain {length} values")
    return [_finite_number(item, f"{name}[{index}]") for index, item in enumerate(items)]


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def improve_policy(
    parameters: PolicyParameters,
    episodes: list[list[TrainingStep]],
    *,
    actor_learning_rate: float = 0.018,
    critic_learning_rate: float = 0.035,
    discount: float = 0.992,
) -> tuple[PolicyParameters, LearningMetrics]:
    """Apply one REINFORCE update with a learned linear value baseline."""
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    if actor_learning_rate <= 0.0 or critic_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")

    samples: list[tuple[TrainingStep, float, float]] = []
    episode_returns: list[float] = []
    for episode in episodes:
        discounted_returns = [0.0 for _ in episode]
        running_return = 0.0
        for index in range(len(episode) - 1, -1, -1):
            running_return = episode[index].reward + discount * running_return
            discounted_returns[index] = running_return
        if discounted_returns:
            episode_returns.append(discounted_returns[0])
        for step, discounted_return in zip(episode, discounted_returns, strict=True):
            value_estimate = _dot(parameters.critic_weights, step.features)
            samples.append((step, discounted_return, discounted_return - value_estimate))

    if not samples:
        return parameters.copy(), LearningMetrics(len(episodes), 0, 0.0, 0.0)

    advantage_mean = sum(sample[2] for sample in samples) / len(samples)
    advantage_variance = sum((sample[2] - advantage_mean) ** 2 for sample in samples) / len(samples)
    advantage_scale = max(1.0e-6, sqrt(advantage_variance))
    actor_gradient = [[0.0 for _ in range(FEATURE_COUNT)] for _ in range(ACTION_COUNT)]
    critic_gradient = [0.0 for _ in range(FEATURE_COUNT)]

    for step, discounted_return, advantage in samples:
        normalized_advantage = _clamp((advantage - advantage_mean) / advantage_scale, -3.0, 3.0)
        for action_index in range(ACTION_COUNT):
            activation = tanh(_dot(parameters.actor_weights[action_index], step.features))
            mean_derivative = _ACTION_RESIDUAL_SCALES[action_index] * (1.0 - activation * activation)
            variance = _EXPLORATION_STANDARD_DEVIATIONS[action_index] ** 2
            log_probability_derivative = step.action_noise[action_index] * mean_derivative / variance
            for feature_index, feature in enumerate(step.features):
                actor_gradient[action_index][feature_index] += (
                    normalized_advantage * log_probability_derivative * feature
                )

        value_error = _clamp(discounted_return - _dot(parameters.critic_weights, step.features), -25.0, 25.0)
        for feature_index, feature in enumerate(step.features):
            critic_gradient[feature_index] += value_error * feature

    sample_scale = 1.0 / len(samples)
    updated = parameters.copy()
    squared_actor_gradient = 0.0
    for action_index in range(ACTION_COUNT):
        for feature_index in range(FEATURE_COUNT):
            gradient = _clamp(actor_gradient[action_index][feature_index] * sample_scale, -4.0, 4.0)
            squared_actor_gradient += gradient * gradient
            updated.actor_weights[action_index][feature_index] += actor_learning_rate * gradient
            updated.actor_weights[action_index][feature_index] = _clamp(
                updated.actor_weights[action_index][feature_index], -3.0, 3.0
            )
    for feature_index in range(FEATURE_COUNT):
        updated.critic_weights[feature_index] += critic_learning_rate * critic_gradient[feature_index] * sample_scale
        updated.critic_weights[feature_index] = _clamp(updated.critic_weights[feature_index], -50.0, 50.0)
    updated.training_iterations += 1

    mean_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
    return updated, LearningMetrics(
        episode_count=len(episodes),
        decision_count=len(samples),
        mean_discounted_return=mean_return,
        actor_gradient_norm=sqrt(squared_actor_gradient),
    )


def _dot(weights: list[float], features: tuple[float, ...]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features, strict=True))


class Controller:
    """Stateful controller used for both deterministic inference and training."""

    def __init__(
        self,
        parameters: PolicyParameters | None = None,
        *,
        training: bool = False,
        random_seed: int = 110,
        planner_config: PlannerConfig | None = None,
        decision_interval_ticks: int = DECISION_INTERVAL_TICKS,
    ) -> None:
        if decision_interval_ticks < 1:
            raise ValueError("decision_interval_ticks must be positive")
        self.parameters = load_policy_parameters() if parameters is None else parameters
        self.training = training
        self.planner_config = PlannerConfig() if planner_config is None else planner_config
        self.decision_interval_ticks = decision_interval_ticks
        self._random = Random(random_seed)
        self._last_command = RobotCommand()
        self._held_command = RobotCommand()
        self._pending_features: tuple[float, ...] | None = None
        self._pending_noise = (0.0, 0.0)
        self._pending_reward = 0.0
        self._previous_damage = 0.0
        self._previous_wall_contact = False
        self._smoothed_target_speed_mps: float | None = None
        self._brake_pulse_active = False
        self._episode: list[TrainingStep] = []

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            self._reset_runtime_state(sensors)

        if self.training and self._pending_features is not None:
            self._pending_reward += self._reward_from_sensors(sensors)

        if sensors.tick % self.decision_interval_ticks != 0:
            return self._held_command

        if self.training:
            self._finish_pending_decision()

        base_command, self._smoothed_target_speed_mps = _geometric_command(
            sensors,
            previous_target_speed_mps=self._smoothed_target_speed_mps,
            config=self.planner_config,
        )
        features = _policy_features(sensors, base_command, self._last_command)
        action_means = _policy_action_means(self.parameters, features, base_command)
        if self.training:
            noises = tuple(
                self._random.gauss(0.0, standard_deviation) for standard_deviation in _EXPLORATION_STANDARD_DEVIATIONS
            )
            throttle = action_means[_ACTION_THROTTLE] + noises[_ACTION_THROTTLE]
            steer = action_means[_ACTION_STEER] + noises[_ACTION_STEER]
            self._pending_features = features
            self._pending_noise = (noises[_ACTION_THROTTLE], noises[_ACTION_STEER])
            self._pending_reward = 0.0
        else:
            throttle = action_means[_ACTION_THROTTLE]
            steer = action_means[_ACTION_STEER]

        # The residual policy may refine acceleration, but it must not cancel
        # a deterministic overspeed request.  Old checkpoints were trained
        # with signed throttle and can otherwise add power into a corner.
        if base_command.throttle <= 0.0:
            throttle = min(throttle, base_command.throttle)

        # A negative command requests reverse, so holding it would latch the
        # drivetrain in braking until the car stopped.  Alternate short brake
        # pulses with zero-throttle decisions: the zero clears that pending
        # direction change while retaining useful service braking.
        if sensors.contact.wall <= 0.0:
            if throttle < 0.0 and not self._brake_pulse_active:
                throttle = max(-1.0, throttle)
                self._brake_pulse_active = True
            else:
                if self._brake_pulse_active:
                    throttle = 0.0
                self._brake_pulse_active = False

        command = RobotCommand(throttle=_clamp(throttle, -1.0, 1.0), steer=_clamp(steer, -1.0, 1.0))
        self._last_command = command
        self._held_command = command
        return command

    def finish_episode(self, terminal_reward: float = 0.0) -> list[TrainingStep]:
        """Finish and return collected training decisions, then clear them."""
        self._pending_reward += terminal_reward
        self._finish_pending_decision()
        episode = self._episode
        self._episode = []
        return episode

    def copy_for_car(self) -> Controller:
        """Return independent runtime state while sharing fixed policy values."""
        return Controller(
            self.parameters.copy(),
            training=False,
            planner_config=self.planner_config,
            decision_interval_ticks=self.decision_interval_ticks,
        )

    def _reset_runtime_state(self, sensors: RobotSensors) -> None:
        self._last_command = RobotCommand()
        self._held_command = RobotCommand()
        self._pending_features = None
        self._pending_noise = (0.0, 0.0)
        self._pending_reward = 0.0
        self._previous_damage = sensors.contact.damage
        self._previous_wall_contact = sensors.contact.wall > 0.0
        self._smoothed_target_speed_mps = None
        self._brake_pulse_active = False
        self._episode = []

    def _finish_pending_decision(self) -> None:
        if self._pending_features is None:
            return
        self._episode.append(TrainingStep(self._pending_features, self._pending_noise, self._pending_reward))
        self._pending_features = None
        self._pending_noise = (0.0, 0.0)
        self._pending_reward = 0.0

    def _reward_from_sensors(self, sensors: RobotSensors) -> float:
        heading_error_radians = radians(sensors.camera.heading_error_degrees) if sensors.camera.visible else 0.0
        forward_progress_proxy = sensors.odometry.speed_mps * max(-0.25, cos(heading_error_radians))
        lane_penalty = 0.06 * abs(sensors.camera.center_offset_m) if sensors.camera.visible else 0.0
        instability_penalty = 0.0015 * abs(sensors.imu.yaw_rate_degrees_per_s)
        control_penalty = 0.012 * self._last_command.steer**2
        reward = sensors.dt_s * (forward_progress_proxy - lane_penalty - instability_penalty - control_penalty)

        damage_increase = max(0.0, sensors.contact.damage - self._previous_damage)
        reward -= 80.0 * damage_increase
        wall_contact = sensors.contact.wall > 0.0
        if wall_contact:
            reward -= 3.0 * sensors.dt_s
        if wall_contact and not self._previous_wall_contact:
            reward -= 0.75
        self._previous_damage = sensors.contact.damage
        self._previous_wall_contact = wall_contact
        return reward


def _policy_action_means(
    parameters: PolicyParameters,
    features: tuple[float, ...],
    base_command: RobotCommand,
) -> tuple[float, float]:
    throttle_residual = _ACTION_RESIDUAL_SCALES[_ACTION_THROTTLE] * tanh(
        _dot(parameters.actor_weights[_ACTION_THROTTLE], features)
    )
    steer_residual = _ACTION_RESIDUAL_SCALES[_ACTION_STEER] * tanh(
        _dot(parameters.actor_weights[_ACTION_STEER], features)
    )
    return (base_command.throttle + throttle_residual, base_command.steer + steer_residual)


def _policy_features(
    sensors: RobotSensors,
    base_command: RobotCommand,
    previous_command: RobotCommand,
) -> tuple[float, ...]:
    lookahead_offsets = _camera_lookahead_offsets(sensors)
    left_wall = _capped_distance(sensors.wall_lidar.left_m, 20.0)
    right_wall = _capped_distance(sensors.wall_lidar.right_m, 20.0)
    front_wall = _capped_distance(sensors.wall_lidar.front_m, 25.0)
    curve_severity = _curve_severity(sensors, lookahead_offsets)
    return (
        1.0,
        _clamp(sensors.odometry.speed_mps / 16.0, -1.0, 1.0),
        _clamp(sensors.camera.center_offset_m / 3.0, -1.5, 1.5),
        _clamp(sensors.camera.heading_error_degrees / 60.0, -1.5, 1.5),
        _clamp(lookahead_offsets[0] / 4.0, -1.5, 1.5),
        _clamp(lookahead_offsets[1] / 9.0, -1.5, 1.5),
        _clamp(lookahead_offsets[2] / 16.0, -1.5, 1.5),
        _clamp(sensors.imu.yaw_rate_degrees_per_s / 140.0, -1.5, 1.5),
        _clamp(sensors.imu.lateral_acceleration_mps2 / 45.0, -1.5, 1.5),
        1.0 - front_wall / 25.0,
        _clamp((left_wall - right_wall) / 20.0, -1.0, 1.0),
        previous_command.throttle,
        previous_command.steer,
        base_command.steer,
        curve_severity,
    )


def _geometric_command(
    sensors: RobotSensors,
    *,
    previous_target_speed_mps: float | None,
    config: PlannerConfig,
) -> tuple[RobotCommand, float | None]:
    """Track the processed camera centerline and brake before visible bends."""
    speed = max(0.0, sensors.odometry.speed_mps)
    if sensors.contact.wall > 0.0:
        open_side_steer = -0.72 if sensors.lidar.left_m > sensors.lidar.right_m else 0.72
        return RobotCommand(throttle=-1.0 if speed > 0.5 else -0.42, steer=open_side_steer), None
    if not sensors.camera.visible:
        return RobotCommand(throttle=0.18, steer=0.0), previous_target_speed_mps

    lookahead_offsets = _camera_lookahead_offsets(sensors)
    lookahead_distance = _clamp(4.0 + speed * 0.72, 4.0, 16.0)
    target_offset = _interpolated_lookahead_offset(lookahead_distance, lookahead_offsets)
    far_curve = _clamp(lookahead_offsets[2] / 8.0, -1.0, 1.0)
    target_offset += config.apex_bias_m * far_curve
    target_angle = atan2(target_offset, lookahead_distance)
    steering_angle = atan(2.0 * _WHEELBASE_M * target_angle / max(2.0, lookahead_distance))
    pure_pursuit_steer = steering_angle / _MAX_STEERING_RADIANS
    heading_term = sensors.camera.heading_error_degrees / 90.0
    center_term = sensors.camera.center_offset_m / 4.0
    yaw_damping = sensors.imu.yaw_rate_degrees_per_s / 1_100.0
    steer = _clamp(
        config.pure_pursuit_gain * pure_pursuit_steer
        + config.heading_gain * heading_term
        + config.center_gain * center_term
        - yaw_damping,
        -1.0,
        1.0,
    )

    curve_severity = _curve_severity(sensors, lookahead_offsets)
    # A ten-second lap of this 181 m track requires more than 18 m/s average,
    # so a 17 m/s target ceiling makes the goal mathematically impossible.
    # Use the far preview to start turning early, retain a useful speed margin
    # in sharp bends, and allow the car to exploit the long straights.
    raw_target_speed = _clamp(
        config.speed_numerator / (1.0 + config.curve_speed_gain * curve_severity),
        config.minimum_corner_speed,
        config.speed_numerator,
    )
    tracking_error = max(
        abs(sensors.camera.center_offset_m) / 2.4,
        abs(sensors.camera.heading_error_degrees) / 55.0,
    )
    if tracking_error > 0.55:
        raw_target_speed /= 1.0 + 0.38 * (tracking_error - 0.55) ** 2
        raw_target_speed = max(10.5, raw_target_speed)

    target_speed = raw_target_speed
    if previous_target_speed_mps is not None:
        # Camera curvature can jump when a corner crosses a lookahead sample.
        # Rate-limit the requested speed instead of weakening the brakes.
        target_speed = _clamp(
            raw_target_speed,
            previous_target_speed_mps - 3.0,
            previous_target_speed_mps + 1.5,
        )
    speed_error = target_speed - speed
    if speed_error < -0.8:
        desired_throttle = _clamp(0.30 * speed_error, -1.0, 0.0)
    elif speed_error > 0.8:
        desired_throttle = _clamp(0.28 + 0.13 * speed_error, 0.0, 1.0)
    else:
        desired_throttle = _clamp(0.08 + 0.08 * speed_error, -0.20, 0.24)

    return RobotCommand(throttle=desired_throttle, steer=steer), target_speed


def _camera_lookahead_offsets(sensors: RobotSensors) -> tuple[float, float, float]:
    values = sensors.camera.lookahead_offsets_m
    if len(values) >= 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    padded = [float(value) for value in values]
    while len(padded) < 3:
        padded.append(padded[-1] if padded else sensors.camera.center_offset_m)
    return (padded[0], padded[1], padded[2])


def _interpolated_lookahead_offset(distance_m: float, offsets: tuple[float, float, float]) -> float:
    if distance_m <= 9.0:
        weight = (distance_m - 4.0) / 5.0
        return offsets[0] + _clamp(weight, 0.0, 1.0) * (offsets[1] - offsets[0])
    weight = (distance_m - 9.0) / 7.0
    return offsets[1] + _clamp(weight, 0.0, 1.0) * (offsets[2] - offsets[1])


def _curve_severity(sensors: RobotSensors, offsets: tuple[float, float, float]) -> float:
    return _clamp(
        max(
            abs(sensors.camera.heading_error_degrees) / 55.0,
            0.62 * abs(offsets[0]) / 4.0,
            0.90 * abs(offsets[1]) / 9.0,
            1.25 * abs(offsets[2]) / 16.0,
        ),
        0.0,
        1.5,
    )


def _capped_distance(distance_m: float, cap_m: float) -> float:
    if distance_m == inf or not isfinite(distance_m):
        return cap_m
    return _clamp(distance_m, 0.0, cap_m)


def create_controller() -> Controller:
    """Create independent state for every car and repeated race."""
    return Controller()


_module_controller: Controller | None = None


def control(sensors: RobotSensors) -> RobotCommand:
    """Support direct function-style use; the runtime prefers the factory."""
    global _module_controller
    if _module_controller is None or sensors.tick == 0:
        _module_controller = create_controller()
    return _module_controller(sensors)
