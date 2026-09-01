"""Fit a bootstrapped terminal value estimate via fitted value iteration.

The MPC controller can only simulate HORIZON ticks (0.33s) ahead per plan --
nowhere near enough to reason through a multi-second turn. This script fits
V(s), an estimate of the best expected discounted future reward reachable
from state s, and `model_based_mpc.py` adds `DISCOUNT**HORIZON *
V(state_after_horizon)` to each candidate's rollout reward so the planner can
weigh what happens past the ticks it can actually simulate.

`step_reward` is imported from dynamics/reward.py -- the exact function the
runtime planner optimizes -- so this value estimate is fit against the same
objective, not an approximation of it.

Bootstrap target, and why it is not simpler: a first version regressed V(s)
toward `step_reward(next_state) + DISCOUNT * V(next_state)` using the
*recorded* next_state from data collection (fitted *policy evaluation*, not
value iteration). That diverged -- the value range grew to roughly [-525, 16]
over 15 rounds and was still trending more negative -- because most collected
transitions come from an exploration policy that crashes and drifts a lot
(median reward -13, worst -118 across the dataset), so bootstrapping under
"whatever the explorer happened to do next" mostly propagates pessimism, not
a signal useful for planning. Real fitted value iteration bootstraps under
the *best available* next state instead: for a grid of candidate actions,
roll each forward through the learned dynamics model, and take the max of
`step_reward(candidate) + DISCOUNT * V(candidate)`. That approximates V* (the
value under an optimal policy) rather than V^explorer, which is what an MPC
terminal-value bootstrap actually wants.

Usage:
    uv run python scripts/train_value_function.py \\
        --dataset artifacts/dynamics_dataset.npz artifacts/dynamics_dataset_turns.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from controllers.dynamics.numpy_model import NumpyDynamicsModel, load_numpy_dynamics_model
from controllers.dynamics.reward import DISCOUNT, step_reward
from controllers.dynamics.value_torch import DEFAULT_HIDDEN_SIZES, ValueMLP, export_numpy_value_model, normalize_state

DEFAULT_OUTPUT = Path("src/controllers/dynamics/value_model.npz")
DEFAULT_DYNAMICS_MODEL = Path("src/controllers/dynamics/dynamics_model.npz")
MIN_STD = 1e-3

# A small, fixed grid of candidate actions used to bootstrap each round's
# target: throttle x steer, spanning the actuator range coarsely enough to
# stay cheap (grid_size states-per-round forward passes through the
# dynamics model, all batched) while still letting "no good action exists
# from here" show up as a genuinely low max.
THROTTLE_GRID = (-0.6, -0.2, 0.2, 0.5, 0.8, 1.0)
STEER_GRID = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)


def load_dataset_states(paths: list[Path]) -> Tensor:
    states_list: list[Tensor] = []
    for path in paths:
        raw = np.load(path)
        states_list.append(torch.from_numpy(raw["states"]).float())
        print(f"  {path}: {states_list[-1].shape[0]} transitions")
    return torch.cat(states_list)


def bootstrap_targets(
    *,
    states: np.ndarray,
    dynamics_model: NumpyDynamicsModel,
    value_model: ValueMLP,
    state_mean: Tensor,
    state_std: Tensor,
) -> Tensor:
    """max over the action grid of step_reward(next_state) + DISCOUNT * V(next_state), batched over all states."""
    best_target = np.full(states.shape[0], -np.inf, dtype=np.float32)
    with torch.inference_mode():
        for throttle in THROTTLE_GRID:
            for steer in STEER_GRID:
                action = np.tile(np.asarray((throttle, steer), dtype=np.float32), (states.shape[0], 1))
                next_state = dynamics_model.predict_next_state(states, action)
                reward = step_reward(next_state)
                next_state_t = torch.from_numpy(next_state).float()
                value = value_model(normalize_state(next_state_t, state_mean=state_mean, state_std=state_std))
                target = reward + DISCOUNT * value.squeeze(-1).numpy()
                best_target = np.maximum(best_target, target)
    return torch.from_numpy(best_target).float()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a bootstrapped terminal value estimate.")
    parser.add_argument("--dataset", type=Path, nargs="+", default=[Path("artifacts/dynamics_dataset.npz")])
    parser.add_argument("--dynamics-model", type=Path, default=DEFAULT_DYNAMICS_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fvi-rounds", type=int, default=12)
    parser.add_argument("--epochs-per-round", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--target-clip", type=float, default=60.0, help="clip bootstrap targets to +/- this value")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    print("loading datasets:")
    states = load_dataset_states(args.dataset)
    dynamics_model = load_numpy_dynamics_model(args.dynamics_model)

    count = states.shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(count, generator=generator)
    val_count = max(1, int(count * args.val_fraction))
    val_indices, train_indices = permutation[:val_count], permutation[val_count:]
    train_states, val_states = states[train_indices], states[val_indices]
    train_count = train_indices.numel()
    print(f"loaded {count} states ({train_count} train / {val_indices.numel()} val)")

    state_mean = train_states.mean(dim=0)
    state_std = train_states.std(dim=0).clamp_min(MIN_STD)

    model = ValueMLP(hidden_sizes=DEFAULT_HIDDEN_SIZES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # A freshly initialized small-weight network outputs near-zero everywhere,
    # so bootstrapping against it now gives the same "V_0 = 0" starting point
    # as the rest of the loop, without a separately special-cased formula.
    train_targets = torch.clamp(
        bootstrap_targets(
            states=train_states.numpy(),
            dynamics_model=dynamics_model,
            value_model=model,
            state_mean=state_mean,
            state_std=state_std,
        ),
        -args.target_clip,
        args.target_clip,
    )

    for fvi_round in range(args.fvi_rounds):
        for _epoch in range(args.epochs_per_round):
            model.train()
            shuffle = torch.randperm(train_count)
            for start in range(0, train_count, args.batch_size):
                batch = shuffle[start : start + args.batch_size]
                optimizer.zero_grad()
                prediction = model(normalize_state(train_states[batch], state_mean=state_mean, state_std=state_std))
                loss = torch.nn.functional.mse_loss(prediction.squeeze(-1), train_targets[batch])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.inference_mode():
            train_value = model(normalize_state(train_states, state_mean=state_mean, state_std=state_std)).squeeze(-1)
            val_value = model(normalize_state(val_states, state_mean=state_mean, state_std=state_std)).squeeze(-1)
        print(
            f"round {fvi_round:2d}  train value [{train_value.min().item():7.2f}, {train_value.max().item():7.2f}]"
            f" mean {train_value.mean().item():7.2f}   val value [{val_value.min().item():7.2f},"
            f" {val_value.max().item():7.2f}] mean {val_value.mean().item():7.2f}"
        )

        new_targets = bootstrap_targets(
            states=train_states.numpy(),
            dynamics_model=dynamics_model,
            value_model=model,
            state_mean=state_mean,
            state_std=state_std,
        )
        train_targets = torch.clamp(new_targets, -args.target_clip, args.target_clip)

    export_numpy_value_model(args.output, model=model, state_mean=state_mean, state_std=state_std)
    print(f"\nsaved value model to {args.output}")


if __name__ == "__main__":
    main()
