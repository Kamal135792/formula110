"""Fit a learned forward-dynamics model on collected (state, action, next_state) transitions.

This is the "does the dynamics model actually fit?" half of the viability
experiment: if held-out one-step prediction error is low relative to the
natural spread of each feature, the reduced state/action representation is
learnable and worth planning against.

Usage:
    uv run python scripts/train_dynamics_model.py --dataset artifacts/dynamics_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from controllers.dynamics.features import STATE_FIELDS
from controllers.dynamics.torch_model import (
    DEFAULT_HIDDEN_SIZES,
    DynamicsMLP,
    DynamicsNormalization,
    export_numpy_dynamics_model,
)

DEFAULT_OUTPUT = Path("src/controllers/dynamics/dynamics_model.npz")


def load_datasets(paths: list[Path]) -> tuple[Tensor, Tensor, Tensor]:
    """Load and concatenate one or more transition datasets (e.g. random-walk + turn-focused)."""
    states_list: list[Tensor] = []
    actions_list: list[Tensor] = []
    next_states_list: list[Tensor] = []
    for path in paths:
        raw = np.load(path)
        states_list.append(torch.from_numpy(raw["states"]).float())
        actions_list.append(torch.from_numpy(raw["actions"]).float())
        next_states_list.append(torch.from_numpy(raw["next_states"]).float())
        print(f"  {path}: {states_list[-1].shape[0]} transitions")
    return torch.cat(states_list), torch.cat(actions_list), torch.cat(next_states_list)


def split_train_val(count: int, *, val_fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    val_count = max(1, int(count * val_fraction))
    return permutation[val_count:], permutation[:val_count]


def epoch_loss(
    *,
    model: DynamicsMLP,
    normalization: DynamicsNormalization,
    states: Tensor,
    actions: Tensor,
    deltas: Tensor,
) -> Tensor:
    normalized_prediction = model(normalization.normalize_state(states), normalization.normalize_action(actions))
    normalized_target = normalization.normalize_delta(deltas)
    return torch.nn.functional.mse_loss(normalized_prediction, normalized_target)


def report_per_feature_error(
    *,
    model: DynamicsMLP,
    normalization: DynamicsNormalization,
    states: Tensor,
    actions: Tensor,
    deltas: Tensor,
) -> None:
    with torch.inference_mode():
        normalized_prediction = model(normalization.normalize_state(states), normalization.normalize_action(actions))
        predicted_delta = normalization.denormalize_delta(normalized_prediction)
    mean_absolute_error = (predicted_delta - deltas).abs().mean(dim=0)
    naive_mean_absolute_error = (deltas - deltas.mean(dim=0, keepdim=True)).abs().mean(dim=0)
    print("\nheld-out one-step delta prediction, model vs naive (predict mean delta) MAE:")
    for index, name in enumerate(STATE_FIELDS):
        model_error = mean_absolute_error[index].item()
        naive_error = naive_mean_absolute_error[index].item()
        print(f"  {name:<26} model {model_error:8.4f}   naive {naive_error:8.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the learned dynamics model.")
    parser.add_argument("--dataset", type=Path, nargs="+", default=[Path("artifacts/dynamics_dataset.npz")])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    print("loading datasets:")
    states, actions, next_states = load_datasets(args.dataset)
    deltas = next_states - states
    train_indices, val_indices = split_train_val(states.shape[0], val_fraction=args.val_fraction, seed=args.seed)
    train_states, train_actions, train_deltas = states[train_indices], actions[train_indices], deltas[train_indices]
    val_states, val_actions, val_deltas = states[val_indices], actions[val_indices], deltas[val_indices]
    print(f"loaded {states.shape[0]} transitions ({train_indices.numel()} train / {val_indices.numel()} val)")

    normalization = DynamicsNormalization.fit(states=train_states, actions=train_actions, deltas=train_deltas)
    model = DynamicsMLP(hidden_sizes=DEFAULT_HIDDEN_SIZES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_count = train_indices.numel()
    for epoch in range(args.epochs):
        model.train()
        shuffle = torch.randperm(train_count)
        running_loss = 0.0
        for start in range(0, train_count, args.batch_size):
            batch_indices = shuffle[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = epoch_loss(
                model=model,
                normalization=normalization,
                states=train_states[batch_indices],
                actions=train_actions[batch_indices],
                deltas=train_deltas[batch_indices],
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_indices.numel()
        train_loss = running_loss / train_count

        model.eval()
        with torch.inference_mode():
            val_loss = epoch_loss(
                model=model, normalization=normalization, states=val_states, actions=val_actions, deltas=val_deltas
            ).item()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")

    model.eval()
    report_per_feature_error(
        model=model, normalization=normalization, states=val_states, actions=val_actions, deltas=val_deltas
    )

    export_numpy_dynamics_model(args.output, model=model, normalization=normalization)
    print(f"\nsaved dynamics model to {args.output}")


if __name__ == "__main__":
    main()
