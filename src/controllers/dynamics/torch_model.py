"""PyTorch training-time dynamics model. Training-only: never imported at runtime.

`scripts/train_dynamics_model.py` is the only caller. It never imports
`racing`, so it does not hit the torch-after-panda3d DLL conflict described in
`dynamics/numpy_model.py`. The trained weights are exported to a plain NumPy
`.npz` file that `dynamics/numpy_model.py` loads for runtime inference.

The model predicts a normalized next-state *delta* rather than the absolute
next state. Delta prediction keeps the regression target close to zero for a
mostly-smooth simulation, which trains faster and generalizes better than
predicting the raw next state directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from controllers.dynamics.features import ACTION_DIM, STATE_DIM

DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (128, 128)
MIN_STD = 1e-3


class DynamicsMLP(nn.Module):
    """Maps normalized (state, action) to a normalized next-state delta."""

    def __init__(self, hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = STATE_DIM + ACTION_DIM
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, STATE_DIM))
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential(*layers)

    def forward(self, normalized_state: Tensor, normalized_action: Tensor) -> Tensor:
        """Return the normalized predicted next-state delta."""
        return self.net(torch.cat((normalized_state, normalized_action), dim=-1))


@dataclass(frozen=True, slots=True)
class DynamicsNormalization:
    """Per-feature mean/std used to normalize model inputs and denormalize outputs."""

    state_mean: Tensor
    state_std: Tensor
    action_mean: Tensor
    action_std: Tensor
    delta_mean: Tensor
    delta_std: Tensor

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: Tensor) -> Tensor:
        return (action - self.action_mean) / self.action_std

    def normalize_delta(self, delta: Tensor) -> Tensor:
        return (delta - self.delta_mean) / self.delta_std

    def denormalize_delta(self, normalized_delta: Tensor) -> Tensor:
        return normalized_delta * self.delta_std + self.delta_mean

    @staticmethod
    def fit(*, states: Tensor, actions: Tensor, deltas: Tensor) -> DynamicsNormalization:
        """Compute normalization statistics from a transition dataset."""
        return DynamicsNormalization(
            state_mean=states.mean(dim=0),
            state_std=states.std(dim=0).clamp_min(MIN_STD),
            action_mean=actions.mean(dim=0),
            action_std=actions.std(dim=0).clamp_min(MIN_STD),
            delta_mean=deltas.mean(dim=0),
            delta_std=deltas.std(dim=0).clamp_min(MIN_STD),
        )


def predict_next_state(
    *,
    model: DynamicsMLP,
    normalization: DynamicsNormalization,
    state: Tensor,
    action: Tensor,
) -> Tensor:
    """Roll `state` forward one tick under `action` using the learned dynamics model.

    Accepts and returns raw (unnormalized) physical-unit tensors, batched on
    the leading dimension. Used during training/validation only; the runtime
    MPC controller uses the NumPy equivalent in `numpy_model.py`.
    """
    normalized_delta = model(normalization.normalize_state(state), normalization.normalize_action(action))
    return state + normalization.denormalize_delta(normalized_delta)


def export_numpy_dynamics_model(
    path: Path,
    *,
    model: DynamicsMLP,
    normalization: DynamicsNormalization,
) -> None:
    """Export trained weights and normalization stats to a runtime-loadable `.npz` file."""
    linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"layer_count": np.asarray(len(linear_layers))}
    for index, layer in enumerate(linear_layers):
        # nn.Linear stores weight as (out, in); the NumPy forward pass computes
        # `activation @ weight`, so transpose to (in, out) once at export time.
        arrays[f"weight_{index}"] = layer.weight.detach().numpy().T.astype(np.float32)
        arrays[f"bias_{index}"] = layer.bias.detach().numpy().astype(np.float32)
    arrays["state_mean"] = normalization.state_mean.numpy().astype(np.float32)
    arrays["state_std"] = normalization.state_std.numpy().astype(np.float32)
    arrays["action_mean"] = normalization.action_mean.numpy().astype(np.float32)
    arrays["action_std"] = normalization.action_std.numpy().astype(np.float32)
    arrays["delta_mean"] = normalization.delta_mean.numpy().astype(np.float32)
    arrays["delta_std"] = normalization.delta_std.numpy().astype(np.float32)
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]  # stub misreads a dynamic key as `allow_pickle`
