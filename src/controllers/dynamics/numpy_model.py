"""NumPy-only forward pass for the learned dynamics model, used at runtime.

Training uses PyTorch (see `dynamics/torch_model.py`), but the runtime
controller reimplements inference in plain NumPy. On this Windows dev
machine, importing `racing` first (which pulls in Panda3D's ~50 native DLLs)
leaves too few process-wide static TLS slots for `torch`'s `c10.dll` to
initialize, so `import torch` after `racing` fails with `OSError: [WinError
1114]`. Since the game entrypoint always imports `racing` before dynamically
loading a student controller module, a controller cannot rely on importing
torch at all. A small MLP forward pass is trivial to reproduce in NumPy, so
inference sidesteps the conflict entirely -- and stays lighter for the 512
MiB controller memory boundary besides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class NumpyDynamicsModel:
    """Weights for a `Linear -> Tanh -> ... -> Linear` MLP, plus I/O normalization."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Roll a batch of states forward one tick under an action batch.

        `state` and `action` are `(batch, state_dim)` / `(batch, action_dim)`
        arrays in raw physical units; the result is `(batch, state_dim)` in
        the same raw units.
        """
        normalized_state = (state - self.state_mean) / self.state_std
        normalized_action = (action - self.action_mean) / self.action_std
        activation = np.concatenate([normalized_state, normalized_action], axis=-1)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            activation = np.tanh(activation @ weight + bias)
        normalized_delta = activation @ self.weights[-1] + self.biases[-1]
        delta = normalized_delta * self.delta_std + self.delta_mean
        return state + delta


def load_numpy_dynamics_model(path: Path) -> NumpyDynamicsModel:
    raw = np.load(path)
    layer_count = int(raw["layer_count"])
    weights = tuple(raw[f"weight_{index}"].astype(np.float32) for index in range(layer_count))
    biases = tuple(raw[f"bias_{index}"].astype(np.float32) for index in range(layer_count))
    return NumpyDynamicsModel(
        weights=weights,
        biases=biases,
        state_mean=raw["state_mean"].astype(np.float32),
        state_std=raw["state_std"].astype(np.float32),
        action_mean=raw["action_mean"].astype(np.float32),
        action_std=raw["action_std"].astype(np.float32),
        delta_mean=raw["delta_mean"].astype(np.float32),
        delta_std=raw["delta_std"].astype(np.float32),
    )
