"""PyTorch training-time terminal value model. Training-only: never imported at runtime.

`scripts/train_value_function.py` is the only caller and never imports
`racing`, so it does not hit the torch-after-panda3d DLL conflict described
in `dynamics/numpy_model.py`. Trained weights export to a plain NumPy `.npz`
that `dynamics/value_numpy.py` loads for runtime inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import Tensor, nn

from controllers.dynamics.features import STATE_DIM

DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (64, 64)


class ValueMLP(nn.Module):
    """Maps a normalized state to a scalar value estimate."""

    def __init__(self, hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = STATE_DIM
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential(*layers)

    def forward(self, normalized_state: Tensor) -> Tensor:
        """Return the estimated value, shape `(batch, 1)`."""
        return self.net(normalized_state)


def normalize_state(state: Tensor, *, state_mean: Tensor, state_std: Tensor) -> Tensor:
    return (state - state_mean) / state_std


def export_numpy_value_model(
    path: Path,
    *,
    model: ValueMLP,
    state_mean: Tensor,
    state_std: Tensor,
) -> None:
    """Export trained weights and state normalization to a runtime-loadable `.npz` file."""
    linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"layer_count": np.asarray(len(linear_layers))}
    for index, layer in enumerate(linear_layers):
        arrays[f"weight_{index}"] = layer.weight.detach().numpy().T.astype(np.float32)
        arrays[f"bias_{index}"] = layer.bias.detach().numpy().astype(np.float32)
    arrays["state_mean"] = state_mean.numpy().astype(np.float32)
    arrays["state_std"] = state_std.numpy().astype(np.float32)
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]  # stub misreads a dynamic key as `allow_pickle`
