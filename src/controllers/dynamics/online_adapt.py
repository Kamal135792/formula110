"""Online (in-race) fine-tuning of the learned dynamics model.

The MPC controller ships with a dynamics model pretrained offline on
exploration data (`scripts/train_dynamics_model.py`). That data covers broad,
somewhat chaotic driving; once the controller is actually racing, its own
trajectory is a much narrower, on-policy slice of state space (its own racing
line, its own speed range) that the offline model may fit less well. This
module lets the model keep learning from the car's own live experience during
a race -- a small NumPy replay buffer plus a hand-rolled forward/backward
pass and Adam update for the same small MLP architecture used offline, so it
never needs to import torch at runtime (see `dynamics/numpy_model.py` for
why that is unsafe in this process).

Normalization statistics (mean/std) stay frozen at their offline-fit values;
only the network weights adapt. That sidesteps a moving-target normalization
problem and is enough to let the model specialize to the car's actual
driving distribution over the course of a race.

Weights live only in this controller instance's memory for the current race
(matching `RobotControllerFactory`: a fresh instance, and fresh adaptation,
is created per car and per repeated race) -- "remembers after the first lap"
means within one continuous drive, not across separate process launches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from controllers.dynamics.numpy_model import NumpyDynamicsModel

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


@dataclass
class _AdamState:
    """Per-parameter Adam moment estimates."""

    first_moment: list[np.ndarray]
    second_moment: list[np.ndarray]
    step: int = 0

    @staticmethod
    def zeros_like(parameters: list[np.ndarray]) -> _AdamState:
        return _AdamState(
            first_moment=[np.zeros_like(p) for p in parameters],
            second_moment=[np.zeros_like(p) for p in parameters],
        )


class AdaptiveDynamicsModel:
    """A copy of a trained `NumpyDynamicsModel` whose weights can keep learning."""

    def __init__(self, base_model: NumpyDynamicsModel, *, learning_rate: float = 5e-4) -> None:
        self.weights = [w.copy() for w in base_model.weights]
        self.biases = [b.copy() for b in base_model.biases]
        self.state_mean = base_model.state_mean
        self.state_std = base_model.state_std
        self.action_mean = base_model.action_mean
        self.action_std = base_model.action_std
        self.delta_mean = base_model.delta_mean
        self.delta_std = base_model.delta_std
        self.learning_rate = learning_rate
        self._adam = _AdamState.zeros_like(self._parameters())

    def _parameters(self) -> list[np.ndarray]:
        return [*self.weights, *self.biases]

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Roll a batch of states forward one tick. Hot path: used every MPC rollout step."""
        normalized_state = (state - self.state_mean) / self.state_std
        normalized_action = (action - self.action_mean) / self.action_std
        activation = np.concatenate([normalized_state, normalized_action], axis=-1)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            activation = np.tanh(activation @ weight + bias)
        normalized_delta = activation @ self.weights[-1] + self.biases[-1]
        delta = normalized_delta * self.delta_std + self.delta_mean
        return state + delta

    def train_step(self, *, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> float:
        """One manual forward/backward/Adam step on a batch of live transitions. Returns the loss."""
        normalized_state = (states - self.state_mean) / self.state_std
        normalized_action = (actions - self.action_mean) / self.action_std
        target = ((next_states - states) - self.delta_mean) / self.delta_std

        activations = [np.concatenate([normalized_state, normalized_action], axis=-1)]
        pre_activations: list[np.ndarray] = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            pre = activations[-1] @ weight + bias
            pre_activations.append(pre)
            activations.append(np.tanh(pre))
        prediction = activations[-1] @ self.weights[-1] + self.biases[-1]

        batch_size, state_dim = target.shape
        element_count = batch_size * state_dim
        error = prediction - target
        loss = float(np.mean(error**2))

        weight_grads: list[np.ndarray] = [np.zeros(0)] * len(self.weights)
        bias_grads: list[np.ndarray] = [np.zeros(0)] * len(self.biases)

        grad_output = 2.0 * error / element_count  # dL/d(prediction), MSE mean over all elements
        weight_grads[-1] = activations[-1].T @ grad_output
        bias_grads[-1] = grad_output.sum(axis=0)
        grad_activation = grad_output @ self.weights[-1].T

        for layer_index in range(len(self.weights) - 2, -1, -1):
            grad_pre = grad_activation * (1.0 - activations[layer_index + 1] ** 2)
            weight_grads[layer_index] = activations[layer_index].T @ grad_pre
            bias_grads[layer_index] = grad_pre.sum(axis=0)
            grad_activation = grad_pre @ self.weights[layer_index].T

        self._apply_adam_step([*weight_grads, *bias_grads])
        return loss

    def _apply_adam_step(self, grads: list[np.ndarray]) -> None:
        self._adam.step += 1
        step = self._adam.step
        bias_correction1 = 1.0 - ADAM_BETA1**step
        bias_correction2 = 1.0 - ADAM_BETA2**step
        parameters = self._parameters()
        for index, (parameter, grad) in enumerate(zip(parameters, grads, strict=True)):
            first_moment = self._adam.first_moment[index]
            second_moment = self._adam.second_moment[index]
            first_moment *= ADAM_BETA1
            first_moment += (1.0 - ADAM_BETA1) * grad
            second_moment *= ADAM_BETA2
            second_moment += (1.0 - ADAM_BETA2) * (grad**2)
            first_moment_hat = first_moment / bias_correction1
            second_moment_hat = second_moment / bias_correction2
            parameter -= self.learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + ADAM_EPS)


@dataclass
class ReplayBuffer:
    """Fixed-capacity ring buffer of live (state, action, next_state) transitions."""

    state_dim: int
    action_dim: int
    capacity: int = 4000
    states: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    next_states: np.ndarray = field(init=False)
    _write_index: int = field(init=False, default=0)
    size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)

    def add(self, *, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        index = self._write_index
        self.states[index] = state
        self.actions[index] = action
        self.next_states[index] = next_state
        self._write_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.integers(0, self.size, size=min(batch_size, self.size))
        return self.states[indices], self.actions[indices], self.next_states[indices]
