"""Model-predictive controller driven by a learned dynamics model.

This is the "planning" half of the model-based learning experiment: instead
of hand-coding a driving policy, `scripts/train_dynamics_model.py` fits a
small MLP that predicts how the reduced state (speed, heading error, wall
ranges, ...) evolves under an action. At every tick this controller searches
for a good action sequence against that learned model and drives with the
first action of the best one (receding-horizon MPC).

Iteration 2 added turn-vs-straightaway awareness: `dynamics/segment.py`
estimates track curvature from `camera.lookahead_offsets_m` alone (no
privileged track state), and the reward blends between a straightaway
profile (loose tracking, biased toward throttle) and a turn profile (tight
heading/center tracking, wider side-wall margin) using that curvature at
every step of the imagined rollout, not just the current tick. Iteration 2
also diagnosed and fixed a stuck-in-the-corner failure mode with an explicit
anti-stall penalty (see the comment above STALL_SPEED_FLOOR_MPS).

Iteration 3 goes three directions at once:

1. Smoother driving. One-shot random shooting picks an independent noisy
   sample each tick, which can flip between similarly-scored but different
   candidates tick to tick (chattering). This switches to a small CEM
   (cross-entropy method) loop: sample, evaluate, narrow the search around
   the elite candidates, repeat for a few iterations -- literally "repeat
   planning until it converges" rather than a single noisy draw. The
   executed action is a reward-weighted (softmax) average of the final
   population rather than a hard argmax, which further smooths tick-to-tick
   output the way MPPI does. A tick-to-tick continuity penalty and an
   in-sequence jerk penalty round this out.
2. Longer horizon. HORIZON went from 15 ticks (0.25s) to 30 (0.5s) so the
   planner can see a turn's exit, not just its entry.
3. Online adaptation. `dynamics/online_adapt.py` lets the dynamics model
   keep fine-tuning on the car's own live driving via a small in-memory
   replay buffer and a hand-rolled NumPy Adam step (verified against
   numerical gradients). The offline-trained model saw broad, chaotic
   exploration data; the car's actual racing line is a narrower, on-policy
   slice it can specialize to as a race goes on. This state lives only in
   this controller instance for the current race -- "remembers lap to lap"
   within one continuous drive, not persisted across separate process runs
   (a fresh instance is created per car and per repeated race, matching
   `RobotControllerFactory`).

Inference is NumPy-only rather than PyTorch; see `dynamics/numpy_model.py`
for why importing torch inside this process is unsafe on this dev machine.

Requires `src/controllers/dynamics/dynamics_model.npz`, produced by:
    uv run python scripts/collect_dynamics_data.py
    uv run python scripts/train_dynamics_model.py
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import threadpoolctl

from controllers.dynamics.features import ACTION_DIM, STATE_DIM, action_to_command, clip_state, sensors_to_state
from controllers.dynamics.numpy_model import load_numpy_dynamics_model
from controllers.dynamics.online_adapt import AdaptiveDynamicsModel, ReplayBuffer
from controllers.dynamics.reward import DISCOUNT, step_reward
from controllers.dynamics.segment import Segment, classify_segment
from racing import RobotCommand, RobotSensors

# NumPy's OpenBLAS backend defaults to spawning up to MAX_THREADS (24 on the
# build used here) worker threads per matmul. Every matmul in this
# controller's rollouts is tiny (a population of a few hundred rows through a
# couple 128-unit layers), so thread spawn/join overhead vastly exceeds the
# actual FLOPs -- measured on this dev machine, that overhead was wildly
# erratic under any background system load: median ~40ms/tick but a p90 of
# 440ms and a max of 473ms, dangerously close to the Gradescope worker's
# 500ms per-call timeout (autograder/gradescope/race_worker.py). Pinning
# OpenBLAS to a single thread removed the erratic tail entirely and was
# *faster* on the (now stable) median too: ~15ms/tick. threadpoolctl (rather
# than setting OMP_NUM_THREADS/OPENBLAS_NUM_THREADS before import) is used
# because it applies the limit at runtime regardless of whether something
# earlier in the host process already imported and initialized NumPy's BLAS.
threadpoolctl.threadpool_limits(1)

# Every planning call allocates a batch of short-lived NumPy arrays
# (candidates, noise, per-step rollout state). Python's cyclic garbage
# collector's default thresholds (700 gen-0 allocations between checks) made
# it sweep often enough under that allocation churn to still show up as tail
# latency after the BLAS fix above (p99 ~158ms, max ~336ms over 300 calls).
# Raising the thresholds -- rather than gc.disable() -- keeps the collector
# available to reclaim genuine reference cycles (this process also runs
# Panda3D's scene graph) while making it run far less often; this measured
# p99 ~32ms, max ~34ms over the same 300 calls.
gc.set_threshold(50_000, 50, 50)

RACING_NAME = "Model-Based MPC"
RACING_COLOR = "#8A2BE2"

MODEL_PATH = Path(__file__).parent / "dynamics" / "dynamics_model.npz"

HORIZON = 20

# CEM (cross-entropy method): sample a population, keep the elite fraction,
# refit the per-step mean/std from them, and repeat. Each iteration narrows
# the search instead of throwing it away, which both improves solution
# quality and (since the mean moves smoothly) keeps the plan from changing
# unpredictably tick to tick.
#
# Per-tick cost is roughly HORIZON x CEM_ITERATIONS x CEM_POPULATION
# sequential batched matmuls, and on this dev machine that cost is noisy
# (median ~25-35ms at these settings, with occasional spikes into the low
# hundreds of ms from what looks like OS-level scheduling jitter, not
# anything data-dependent -- state divergence was a real bug worth fixing
# in its own right, see clip_state, but it did not explain the spikes). The
# first version of this (HORIZON=30, 3 iterations, population 160) measured
# ~92ms/tick median; that is still far under the autograder's 500ms hard
# limit but would visibly slow-motion an interactive --watch race, since
# each control() call gates one fixed physics tick. These settings trade
# some of the horizon/iteration/population budget down to keep the median
# closer to real time while still meaningfully repeating the search
# (CEM_ITERATIONS) and looking further ahead (HORIZON) than the first
# single-shot version (HORIZON=15, one iteration, population 200, ~8ms/tick).
CEM_ITERATIONS = 2
CEM_POPULATION = 120
CEM_ELITE_FRACTION = 0.15
CEM_MIN_STD = np.asarray((0.06, 0.06), dtype=np.float32)
# How concentrated the final reward-weighted average is on the best
# candidates; scaled per-tick by that tick's reward spread so it adapts to
# whatever the reward's natural scale happens to be.
CEM_TEMPERATURE_SCALE = 0.5

# Initial sampling noise and throttle bias, per current segment: straightaways
# get less steer noise and a forward throttle push; turns get more steer
# noise (to explore sharper turn-in) and a lighter throttle push. CEM shrinks
# these each iteration, so they only set the starting spread.
STRAIGHT_THROTTLE_NOISE_STD = 0.25
STRAIGHT_STEER_NOISE_STD = 0.20
STRAIGHT_THROTTLE_BIAS = 0.15
TURN_THROTTLE_NOISE_STD = 0.30
TURN_STEER_NOISE_STD = 0.45
TURN_THROTTLE_BIAS = -0.05

# Smoothness: penalize the plan's first action jumping away from what was
# actually applied last tick (tick-to-tick continuity), and penalize large
# step-to-step swings within a candidate's own action sequence (in-plan
# jerk). Both act directly on the sampled actions, not the rolled-out state.
ACTION_JUMP_WEIGHT = 0.8
ACTION_JERK_WEIGHT = 0.3

# The per-tick reward itself (progress/tracking/safety/anti-stall, all
# segment-blended) lives in dynamics/reward.py.
#
# Tried and shelved: a learned terminal value estimate, to let the planner
# weigh what happens past HORIZON without simulating further (dynamics/
# value_torch.py, dynamics/value_numpy.py, scripts/train_value_function.py
# are the working, gradient-checked infrastructure for this -- just not
# wired in here). Diagnosed the stuck-in-a-hairpin failure below as coming
# from two places: a dynamics model whose yaw-rate prediction barely beat
# "assume no change," and a horizon too short to see through a multi-second
# turn. Addressed both: collected a second dataset from an explorer that
# commits to sustained hard steering through detected turns instead of a
# per-tick random walk (see collect_dynamics_data.py's TurnFocusedExplorer),
# and fit a bootstrapped terminal value via fitted value iteration on the
# merged data. The turn-focused data measurably improved yaw-rate prediction
# (MAE 3.13 vs a naive baseline's 3.21, up from roughly tied before) but
# measurably *worsened* wall-distance prediction (e.g. wall_left MAE 0.071
# vs naive 0.058, worse than the single-dataset model's 0.071 vs 0.058 --
# domination on one dimension traded against another). On the standard
# evaluation, merging that data into the shipped dynamics model alone (value
# term at zero) pushed damage from 0.1% to 12.5% and flipped the race record
# from 3-2 to 1-3-1 -- a real regression in the property (near-zero damage)
# that was the whole point of this controller. Adding the terminal value
# back on top partially offset that (6.2% damage) but never got below the
# single-dataset model's own numbers, and the hairpin itself still was not
# cleanly solved either way -- it changed failure mode (from frozen in place
# to slowly drifting toward the wall over several seconds without crashing)
# rather than resolving. Reverted the dynamics model to the original
# single-dataset training and left the terminal value unwired rather than
# ship an unvalidated regression. A next attempt at the underlying model
# weakness should probably use a larger network or targeted data reweighting
# instead of a flat merge, so gains on one state dimension stop coming at
# another's expense.

# Online adaptation: every ADAPT_INTERVAL_TICKS, run ADAPT_STEPS_PER_TRIGGER
# gradient steps on a batch sampled from the car's own recent live
# transitions, once there is enough in the buffer to sample a full batch from
# without heavy repetition. A small learning rate and infrequent, few-step
# updates keep this a gentle specialization rather than a destabilizing
# rewrite of the offline-trained model.
ADAPT_INTERVAL_TICKS = 90
ADAPT_STEPS_PER_TRIGGER = 3
ADAPT_BATCH_SIZE = 64
ADAPT_MIN_BUFFER_SIZE = 256
ADAPT_LEARNING_RATE = 3e-4
REPLAY_BUFFER_CAPACITY = 6000


class ModelBasedMPCController:
    """Receding-horizon CEM planner over a dynamics model that keeps learning online."""

    def __init__(self, *, model_path: Path = MODEL_PATH, seed: int = 110) -> None:
        base_model = load_numpy_dynamics_model(model_path)
        self._model = AdaptiveDynamicsModel(base_model, learning_rate=ADAPT_LEARNING_RATE)
        self._replay_buffer = ReplayBuffer(state_dim=STATE_DIM, action_dim=ACTION_DIM, capacity=REPLAY_BUFFER_CAPACITY)
        self._rng = np.random.default_rng(seed)
        self._warm_start_plan = self._default_plan()
        self._last_applied_action = self._warm_start_plan[0].copy()
        self._pending_transition: tuple[np.ndarray, np.ndarray, int] | None = None
        self._ticks_since_adapt = 0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        state = sensors_to_state(sensors)
        self._record_transition(sensors=sensors, state=state)
        self._maybe_adapt()

        segment = classify_segment(sensors)
        plan = self._plan(state=state, segment=segment)
        self._warm_start_plan = plan
        self._last_applied_action = plan[0].copy()
        self._pending_transition = (state, plan[0].copy(), sensors.tick)
        return action_to_command(plan[0])

    def copy_for_car(self) -> ModelBasedMPCController:
        """Give every car and repeated race its own planner state, model copy, and RNG stream."""
        return ModelBasedMPCController()

    def _default_plan(self) -> np.ndarray:
        plan = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
        plan[:, 0] = 0.4  # mild forward throttle, no steer
        return plan

    def _record_transition(self, *, sensors: RobotSensors, state: np.ndarray) -> None:
        if self._pending_transition is None:
            return
        previous_state, previous_action, previous_tick = self._pending_transition
        if sensors.tick == previous_tick + 1:
            self._replay_buffer.add(state=previous_state, action=previous_action, next_state=state)

    def _maybe_adapt(self) -> None:
        self._ticks_since_adapt += 1
        if self._ticks_since_adapt < ADAPT_INTERVAL_TICKS:
            return
        if self._replay_buffer.size < ADAPT_MIN_BUFFER_SIZE:
            return
        self._ticks_since_adapt = 0
        for _ in range(ADAPT_STEPS_PER_TRIGGER):
            states, actions, next_states = self._replay_buffer.sample(ADAPT_BATCH_SIZE, rng=self._rng)
            self._model.train_step(states=states, actions=actions, next_states=next_states)

    def _plan(self, *, state: np.ndarray, segment: Segment) -> np.ndarray:
        if segment == "straight":
            std = np.asarray((STRAIGHT_THROTTLE_NOISE_STD, STRAIGHT_STEER_NOISE_STD), dtype=np.float32)
            throttle_bias = STRAIGHT_THROTTLE_BIAS
        else:
            std = np.asarray((TURN_THROTTLE_NOISE_STD, TURN_STEER_NOISE_STD), dtype=np.float32)
            throttle_bias = TURN_THROTTLE_BIAS

        mean = np.concatenate((self._warm_start_plan[1:], self._warm_start_plan[-1:]), axis=0).copy()
        mean[:, 0] = np.clip(mean[:, 0] + throttle_bias, -1.0, 1.0)
        std = np.broadcast_to(std, (HORIZON, ACTION_DIM)).copy()

        elite_count = max(2, int(CEM_POPULATION * CEM_ELITE_FRACTION))
        candidates = mean  # placeholder so the type checker sees a definite binding
        reward = np.zeros(1, dtype=np.float32)
        for _iteration in range(CEM_ITERATIONS):
            noise = self._rng.normal(0.0, 1.0, size=(CEM_POPULATION, HORIZON, ACTION_DIM)).astype(np.float32)
            candidates = np.clip(mean[np.newaxis, :, :] + noise * std[np.newaxis, :, :], -1.0, 1.0)
            reward = np.nan_to_num(self._rollout_reward(state=state, candidates=candidates), nan=-1e9)
            elite_indices = np.argpartition(reward, -elite_count)[-elite_count:]
            elite = candidates[elite_indices]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), CEM_MIN_STD)

        # Reward-weighted (softmax) average of the final population, rather
        # than a hard argmax, so the executed plan does not hinge on a single
        # noisy sample -- this is the main lever against tick-to-tick jitter.
        temperature = max(1e-3, float(reward.std()) * CEM_TEMPERATURE_SCALE)
        weights = np.exp((reward - reward.max()) / temperature)
        weights /= weights.sum()
        return np.tensordot(weights, candidates, axes=(0, 0)).astype(np.float32)

    def _rollout_reward(self, *, state: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        num_candidates = candidates.shape[0]
        rollout_state = np.repeat(state[np.newaxis, :], num_candidates, axis=0)
        total_reward = np.zeros(num_candidates, dtype=np.float32)
        discount = 1.0
        for step in range(HORIZON):
            action = candidates[:, step, :]
            next_state = clip_state(self._model.predict_next_state(rollout_state, action))
            total_reward += discount * step_reward(next_state)
            rollout_state = next_state
            discount *= DISCOUNT

        first_step_jump = candidates[:, 0, :] - self._last_applied_action[np.newaxis, :]
        total_reward -= ACTION_JUMP_WEIGHT * (first_step_jump**2).sum(axis=1)
        action_deltas = candidates[:, 1:, :] - candidates[:, :-1, :]
        total_reward -= ACTION_JERK_WEIGHT * (action_deltas**2).sum(axis=(1, 2))
        return total_reward


def create_controller() -> ModelBasedMPCController:
    return ModelBasedMPCController()
