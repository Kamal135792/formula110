"""The MPC reward, factored out so the offline value-function trainer can reuse it exactly.

`scripts/train_value_function.py` fits a bootstrapped terminal value estimate
by regressing V(s) toward `step_reward(next_state) + DISCOUNT * V(next_state)`
over the collected transition dataset. That target is only meaningful if it
is the *same* reward the runtime planner optimizes -- hence living here
instead of inside `model_based_mpc.py`, so both import one definition rather
than risking the two drifting apart.
"""

from __future__ import annotations

import numpy as np

from controllers.dynamics.features import (
    IDX_CENTER_OFFSET_M,
    IDX_HEADING_ERROR_DEG,
    IDX_LOOKAHEAD_OFFSET_2_M,
    IDX_SPEED_MPS,
    IDX_WALL_CONTACT,
    IDX_WALL_FRONT_LEFT_M,
    IDX_WALL_FRONT_M,
    IDX_WALL_FRONT_RIGHT_M,
)
from controllers.dynamics.segment import curvature_degrees_per_m_from_offset, turn_blend_weight

DT_S = 1.0 / 60.0
DISCOUNT = 0.97

# Reward weights blend continuously between these two profiles at every
# rollout step, based on that step's own predicted curvature.
#
# Iteration 3 (of the reward, before CEM/horizon/online-adaptation/terminal-
# value work) tried scaling progress reward up on straights (tried 6x, then a
# more conservative 2x) to close the speed gap against the hand-written
# baselines. Both made things *worse* on every axis -- slower (max speed fell
# further, from 12.6 to 11.5 to 8.7 m/s across no-boost/6x/2x) and less safe
# (marshal resets and low-progress time both rose). That is the opposite of
# ordinary reward-shaping intuition, which means the planner is not just
# "playing it safe" -- boosting one reward term shifts which candidate wins
# the argmax over a chaotic multi-step rollout in ways a single scalar weight
# does not control cleanly, and the learned model's known weak spot (it
# underpredicts wall_contact; see scripts/train_dynamics_model.py's held-out
# MAE report) means a shifted decision boundary can plan through contact the
# model does not see coming. Reverted to a flat, unweighted progress term.
STRAIGHT_PROGRESS_WEIGHT = 1.0
TURN_PROGRESS_WEIGHT = 1.0
STRAIGHT_HEADING_WEIGHT = 0.3
TURN_HEADING_WEIGHT = 1.2
STRAIGHT_CENTER_WEIGHT = 0.3
TURN_CENTER_WEIGHT = 1.0
STRAIGHT_SIDE_SAFE_M = 1.0
TURN_SIDE_SAFE_M = 1.6
STRAIGHT_WALL_WEIGHT = 1.5
TURN_WALL_WEIGHT = 2.5

# Front safety margin scales with predicted speed (time headway) instead of a
# fixed distance, so it is not overly conservative at low speed or
# under-cautious at high speed.
FRONT_REACTION_TIME_S = 0.45
MIN_FRONT_SAFE_M = 1.5
MAX_FRONT_SAFE_M = 6.0

CONTACT_PENALTY = 5.0

# Diagnosed by tracing live telemetry (tick-by-tick speed/heading/center/wall
# readings) through a race: the car was getting stuck crawling in place mid-
# corner -- well-centered (offset 0.1-0.2m), well-aligned (heading error
# 2-7deg), no wall within many meters on any side, yet speed kept bouncing
# between 0 and ~1.5 m/s and never built up. With TURN_HEADING_WEIGHT/
# TURN_CENTER_WEIGHT several times TURN_PROGRESS_WEIGHT, and a short horizon
# too short to see past a turn's entry, "stay still and centered" is a
# genuine local reward optimum: progress reward at near-zero speed is too
# small to outweigh even a small predicted tracking blip from committing to
# the turn. An explicit floor-speed penalty breaks that trap directly,
# instead of reweighting progress globally (which had unpredictable side
# effects elsewhere on the track).
#
# The floor is itself segment-blended: a fixed 1.5 m/s floor also fought a
# genuinely tight corner (curvature 4-6 deg/m, i.e. roughly a 9-15m turn
# radius), where slowing well below that is the physically correct thing to
# do, not stalling. Testing at a longer race duration (25s instead of the
# 12-20s used to diagnose and fix the first stall) found the car still
# freezing near-zero speed at such a corner -- with a uniform floor, the
# reward has no way to tell "idling on an open straight" from "correctly
# crawling through a hairpin" apart. Lowering the floor as curvature rises
# keeps the anti-stall behavior where it is needed (straights, gentle turns)
# without fighting deliberately slow driving through a sharp one.
STRAIGHT_STALL_SPEED_FLOOR_MPS = 1.5
TURN_STALL_SPEED_FLOOR_MPS = 0.4
STALL_PENALTY_WEIGHT = 1.5


def step_reward(next_state: np.ndarray) -> np.ndarray:
    """Per-tick reward for landing in `next_state`. Batched: `next_state` is `(B, STATE_DIM)`."""
    speed_mps = next_state[:, IDX_SPEED_MPS]
    heading_error_deg = next_state[:, IDX_HEADING_ERROR_DEG]
    center_offset_m = next_state[:, IDX_CENTER_OFFSET_M]
    lookahead_far_m = next_state[:, IDX_LOOKAHEAD_OFFSET_2_M]
    front_m = np.clip(next_state[:, IDX_WALL_FRONT_M], 0.0, None)
    front_left_m = np.clip(next_state[:, IDX_WALL_FRONT_LEFT_M], 0.0, None)
    front_right_m = np.clip(next_state[:, IDX_WALL_FRONT_RIGHT_M], 0.0, None)
    wall_contact = np.clip(next_state[:, IDX_WALL_CONTACT], 0.0, 1.0)

    # Blend reward weights toward the turn profile using this state's own
    # predicted curvature -- not the segment at planning time -- so a turn
    # entered mid-horizon is already penalized/rewarded correctly.
    curvature_deg_per_m = curvature_degrees_per_m_from_offset(lookahead_far_m)
    turn_weight = turn_blend_weight(curvature_deg_per_m)
    progress_weight = STRAIGHT_PROGRESS_WEIGHT + turn_weight * (TURN_PROGRESS_WEIGHT - STRAIGHT_PROGRESS_WEIGHT)
    heading_weight = STRAIGHT_HEADING_WEIGHT + turn_weight * (TURN_HEADING_WEIGHT - STRAIGHT_HEADING_WEIGHT)
    center_weight = STRAIGHT_CENTER_WEIGHT + turn_weight * (TURN_CENTER_WEIGHT - STRAIGHT_CENTER_WEIGHT)
    side_safe_m = STRAIGHT_SIDE_SAFE_M + turn_weight * (TURN_SIDE_SAFE_M - STRAIGHT_SIDE_SAFE_M)
    wall_weight = STRAIGHT_WALL_WEIGHT + turn_weight * (TURN_WALL_WEIGHT - STRAIGHT_WALL_WEIGHT)
    stall_speed_floor_mps = STRAIGHT_STALL_SPEED_FLOOR_MPS + turn_weight * (
        TURN_STALL_SPEED_FLOOR_MPS - STRAIGHT_STALL_SPEED_FLOOR_MPS
    )

    front_safe_m = np.clip(speed_mps * FRONT_REACTION_TIME_S, MIN_FRONT_SAFE_M, MAX_FRONT_SAFE_M)

    progress = progress_weight * speed_mps * DT_S
    heading_penalty = heading_weight * (heading_error_deg / 30.0) ** 2
    center_penalty = center_weight * (center_offset_m / 2.0) ** 2
    front_penalty = wall_weight * np.clip(front_safe_m - front_m, 0.0, None)
    side_penalty = 0.5 * wall_weight * (
        np.clip(side_safe_m - front_left_m, 0.0, None) + np.clip(side_safe_m - front_right_m, 0.0, None)
    )
    contact_penalty = CONTACT_PENALTY * wall_contact
    stall_penalty = STALL_PENALTY_WEIGHT * np.clip(stall_speed_floor_mps - speed_mps, 0.0, None) ** 2

    return progress - heading_penalty - center_penalty - front_penalty - side_penalty - contact_penalty - stall_penalty
