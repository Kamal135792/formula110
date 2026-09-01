"""NumPy-only forward pass for the terminal value estimate, used at runtime.

Mirrors `dynamics/numpy_model.py`: training happens in PyTorch
(`dynamics/value_torch.py`, used only by `scripts/train_value_function.py`,
which never imports `racing`), and the runtime controller loads a plain
NumPy `.npz` export so it never imports torch (see `numpy_model.py`'s
docstring for why that import order is unsafe in this process).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class NumpyValueModel:
    """Weights for a `Linear -> Tanh -> ... -> Linear` MLP predicting a scalar value."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    state_mean: np.ndarray
    state_std: np.ndarray

    def predict_value(self, state: np.ndarray) -> np.ndarray:
        """Estimate expected discounted future reward from a batch of states, `(B, state_dim) -> (B,)`."""
        activation = (state - self.state_mean) / self.state_std
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            activation = np.tanh(activation @ weight + bias)
        value = activation @ self.weights[-1] + self.biases[-1]
        return value[:, 0]


def load_numpy_value_model(path: Path) -> NumpyValueModel:
    raw = np.load(path)
    layer_count = int(raw["layer_count"])
    weights = tuple(raw[f"weight_{index}"].astype(np.float32) for index in range(layer_count))
    biases = tuple(raw[f"bias_{index}"].astype(np.float32) for index in range(layer_count))
    return NumpyValueModel(
        weights=weights,
        biases=biases,
        state_mean=raw["state_mean"].astype(np.float32),
        state_std=raw["state_std"].astype(np.float32),
    )
