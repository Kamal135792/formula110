"""Collect (state, action, next_state) transitions for learned-dynamics training.

Runs headless head-to-head races with two independent exploratory controllers
so the physics simulator produces real driving trajectories -- including near
wall contact -- without a human at the wheel. Two exploration strategies are
available:

- "random-walk" (default): a smoothed random walk in action space (with a
  wall-recovery reflex so it does not get stuck grinding against a barrier
  for an entire race), giving broad state/action coverage.
- "turn-focused": reacts to live curvature (dynamics/segment.py) and commits
  to a strong, sustained steering input for 15-35 ticks at a time whenever it
  detects a turn, instead of a fast per-tick random walk. The random-walk
  explorer mostly produces short, noisy taps at any one action, which is a
  poor way to observe how yaw rate actually evolves under *sustained*
  cornering -- exactly what a first evaluation of the trained dynamics model
  found weakest (see scripts/train_dynamics_model.py's held-out MAE report:
  yaw_rate barely beat a naive "predict no change" baseline). This strategy
  trades broad coverage for dense, temporally coherent turning data.

Usage:
    uv run python scripts/collect_dynamics_data.py --output artifacts/dynamics_dataset.npz
    uv run python scripts/collect_dynamics_data.py --strategy turn-focused \\
        --output artifacts/dynamics_dataset_turns.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from controllers.dynamics.features import ACTION_DIM, STATE_DIM, command_to_action, sensors_to_state
from controllers.dynamics.segment import TURN_CURVATURE_THRESHOLD_DEG_PER_M, estimate_curvature_degrees_per_m
from racing import RobotCommand, RobotController, RobotSensors, run_headless_head_to_head
from racing.race.rules import HeadToHeadRaceRules

DEFAULT_SEEDS: tuple[int, ...] = (42, 110, 271, 997, 2027)
WALL_CONTACT_RECOVERY_SECONDS = 0.4
Strategy = Literal["random-walk", "turn-focused"]


class ExplorerController:
    """Smoothed random-walk driver that also records transitions as it drives."""

    def __init__(self, *, seed: int, transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self._rng = np.random.default_rng(seed)
        self._throttle = 0.4
        self._steer = 0.0
        self._transitions = transitions
        self._pending: tuple[np.ndarray, np.ndarray, int] | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        state = sensors_to_state(sensors)
        if self._pending is not None:
            previous_state, previous_action, previous_tick = self._pending
            if sensors.tick == previous_tick + 1:
                self._transitions.append((previous_state, previous_action, state))
        command = self._next_command(sensors)
        self._pending = (state, command_to_action(command), sensors.tick)
        return command

    def _next_command(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.contact.wall > WALL_CONTACT_RECOVERY_SECONDS:
            open_side = -1.0 if sensors.wall_lidar.left_m > sensors.wall_lidar.right_m else 1.0
            self._throttle = -0.6
            self._steer = 0.6 * open_side
        else:
            self._throttle += float(self._rng.normal(0.0, 0.10)) + (0.45 - self._throttle) * 0.03
            self._steer += float(self._rng.normal(0.0, 0.12)) + (0.0 - self._steer) * 0.05
            self._throttle = float(np.clip(self._throttle, -0.4, 1.0))
            self._steer = float(np.clip(self._steer, -1.0, 1.0))
        return RobotCommand(throttle=self._throttle, steer=self._steer)


class TurnFocusedExplorer:
    """Commits to sustained, decisive steering through detected turns instead of a per-tick random walk."""

    def __init__(self, *, seed: int, transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self._rng = np.random.default_rng(seed)
        self._throttle = 0.35
        self._steer = 0.0
        self._transitions = transitions
        self._pending: tuple[np.ndarray, np.ndarray, int] | None = None
        self._commit_ticks_remaining = 0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        state = sensors_to_state(sensors)
        if self._pending is not None:
            previous_state, previous_action, previous_tick = self._pending
            if sensors.tick == previous_tick + 1:
                self._transitions.append((previous_state, previous_action, state))
        command = self._next_command(sensors)
        self._pending = (state, command_to_action(command), sensors.tick)
        return command

    def _next_command(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.contact.wall > WALL_CONTACT_RECOVERY_SECONDS:
            open_side = -1.0 if sensors.wall_lidar.left_m > sensors.wall_lidar.right_m else 1.0
            self._throttle = -0.6
            self._steer = 0.6 * open_side
            self._commit_ticks_remaining = 0
            return RobotCommand(throttle=self._throttle, steer=self._steer)

        if self._commit_ticks_remaining <= 0:
            # Start committing a little before the MPC's own "turn" threshold,
            # so this also densely covers the entry/transition region.
            curvature = estimate_curvature_degrees_per_m(sensors)
            approaching_turn = abs(curvature) >= TURN_CURVATURE_THRESHOLD_DEG_PER_M * 0.5
            if approaching_turn:
                sign = 1.0 if curvature > 0.0 else -1.0
                self._steer = sign * float(self._rng.uniform(0.5, 1.0))
                self._throttle = float(self._rng.uniform(-0.1, 0.7))
                self._commit_ticks_remaining = int(self._rng.integers(15, 35))
            else:
                self._steer = float(self._rng.uniform(-0.15, 0.15))
                self._throttle = float(self._rng.uniform(0.2, 0.7))
                self._commit_ticks_remaining = int(self._rng.integers(10, 25))
        self._commit_ticks_remaining -= 1
        return RobotCommand(throttle=self._throttle, steer=self._steer)


def _make_explorer(
    strategy: Strategy, *, seed: int, transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> RobotController:
    if strategy == "turn-focused":
        return TurnFocusedExplorer(seed=seed, transitions=transitions)
    return ExplorerController(seed=seed, transitions=transitions)


def collect_seed(
    *, seed: int, races: int, round_seconds: float, strategy: Strategy
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    challenger = _make_explorer(strategy, seed=seed * 2 + 1, transitions=transitions)
    incumbent = _make_explorer(strategy, seed=seed * 2 + 2, transitions=transitions)
    run_headless_head_to_head(
        challenger_controller=challenger,
        incumbent_controller=incumbent,
        race_count=races,
        round_seconds=round_seconds,
        random_seed=seed,
        rules=HeadToHeadRaceRules(marshal_enabled=False),
    )
    return transitions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect transitions for learned-dynamics training.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dynamics_dataset.npz"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--races-per-seed", type=int, default=2)
    parser.add_argument("--round-seconds", type=float, default=25.0)
    parser.add_argument("--strategy", choices=("random-walk", "turn-focused"), default="random-walk")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    all_transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for seed in args.seeds:
        seed_transitions = collect_seed(
            seed=int(seed), races=int(args.races_per_seed), round_seconds=float(args.round_seconds),
            strategy=args.strategy,
        )
        print(f"seed {seed}: collected {len(seed_transitions)} transitions")
        all_transitions.extend(seed_transitions)

    if len(all_transitions) == 0:
        raise RuntimeError("no transitions were collected")

    states = np.stack([state for state, _action, _next_state in all_transitions]).astype(np.float32)
    actions = np.stack([action for _state, action, _next_state in all_transitions]).astype(np.float32)
    next_states = np.stack([next_state for _state, _action, next_state in all_transitions]).astype(np.float32)
    assert states.shape == (len(all_transitions), STATE_DIM)
    assert actions.shape == (len(all_transitions), ACTION_DIM)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        states=states,
        actions=actions,
        next_states=next_states,
        seeds=np.asarray(args.seeds, dtype=np.int64),
    )
    print(f"saved {len(all_transitions)} transitions to {args.output}")


if __name__ == "__main__":
    main()
