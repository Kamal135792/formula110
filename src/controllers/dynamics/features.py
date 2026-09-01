"""Sensor/action <-> fixed-size vector conversion shared by data collection, training, and MPC.

The learned dynamics model predicts how this reduced state evolves under an
action, not the full simulator state. Keeping the state small and physically
meaningful (speed, turn rate, track-relative pose, wall ranges) keeps the
first viability experiment tractable on CPU.
"""

from __future__ import annotations

import numpy as np

from racing import RobotCommand, RobotSensors

STATE_FIELDS: tuple[str, ...] = (
    "speed_mps",
    "yaw_rate_degrees_per_s",
    "heading_error_degrees",
    "center_offset_m",
    "lookahead_offset_0_m",
    "lookahead_offset_1_m",
    "lookahead_offset_2_m",
    "wall_front_m",
    "wall_front_left_m",
    "wall_front_right_m",
    "wall_left_m",
    "wall_right_m",
    "wall_contact",
)
ACTION_FIELDS: tuple[str, ...] = ("throttle", "steer")

STATE_DIM = len(STATE_FIELDS)
ACTION_DIM = len(ACTION_FIELDS)

# Index constants so planning/reward code can address state dims by name
# instead of magic numbers, while still working with plain tensors.
IDX_SPEED_MPS = STATE_FIELDS.index("speed_mps")
IDX_YAW_RATE_DPS = STATE_FIELDS.index("yaw_rate_degrees_per_s")
IDX_HEADING_ERROR_DEG = STATE_FIELDS.index("heading_error_degrees")
IDX_CENTER_OFFSET_M = STATE_FIELDS.index("center_offset_m")
IDX_LOOKAHEAD_OFFSET_0_M = STATE_FIELDS.index("lookahead_offset_0_m")
IDX_LOOKAHEAD_OFFSET_1_M = STATE_FIELDS.index("lookahead_offset_1_m")
IDX_LOOKAHEAD_OFFSET_2_M = STATE_FIELDS.index("lookahead_offset_2_m")
IDX_WALL_FRONT_M = STATE_FIELDS.index("wall_front_m")
IDX_WALL_FRONT_LEFT_M = STATE_FIELDS.index("wall_front_left_m")
IDX_WALL_FRONT_RIGHT_M = STATE_FIELDS.index("wall_front_right_m")
IDX_WALL_LEFT_M = STATE_FIELDS.index("wall_left_m")
IDX_WALL_RIGHT_M = STATE_FIELDS.index("wall_right_m")
IDX_WALL_CONTACT = STATE_FIELDS.index("wall_contact")

# Clip caps guard against inf LiDAR no-hits and rare physics outliers. These
# are sanity bounds, not normalization; per-feature mean/std normalization is
# computed from the collected dataset in dynamics/model.py.
SPEED_CAP_MPS = 20.0
YAW_RATE_CAP_DPS = 400.0
OFFSET_CAP_M = 8.0
LIDAR_CAP_M = 25.0


def sensors_to_state(sensors: RobotSensors) -> np.ndarray:
    """Encode one `RobotSensors` snapshot as a fixed-size float32 state vector."""
    lookahead = sensors.camera.lookahead_offsets_m
    lookahead_0 = lookahead[0] if len(lookahead) > 0 else 0.0
    lookahead_1 = lookahead[1] if len(lookahead) > 1 else 0.0
    lookahead_2 = lookahead[2] if len(lookahead) > 2 else 0.0
    values = (
        _clip(sensors.odometry.speed_mps, SPEED_CAP_MPS),
        _clip(sensors.imu.yaw_rate_degrees_per_s, YAW_RATE_CAP_DPS),
        sensors.camera.heading_error_degrees,
        _clip(sensors.camera.center_offset_m, OFFSET_CAP_M),
        _clip(lookahead_0, OFFSET_CAP_M),
        _clip(lookahead_1, OFFSET_CAP_M),
        _clip(lookahead_2, OFFSET_CAP_M),
        _clip_lidar(sensors.wall_lidar.front_m),
        _clip_lidar(sensors.wall_lidar.front_left_m),
        _clip_lidar(sensors.wall_lidar.front_right_m),
        _clip_lidar(sensors.wall_lidar.left_m),
        _clip_lidar(sensors.wall_lidar.right_m),
        1.0 if sensors.contact.wall > 0.0 else 0.0,
    )
    return np.asarray(values, dtype=np.float32)


_SYMMETRIC_CLIP_CAPS: tuple[tuple[int, float], ...] = (
    (IDX_SPEED_MPS, SPEED_CAP_MPS),
    (IDX_YAW_RATE_DPS, YAW_RATE_CAP_DPS),
    (IDX_CENTER_OFFSET_M, OFFSET_CAP_M),
    (IDX_LOOKAHEAD_OFFSET_0_M, OFFSET_CAP_M),
    (IDX_LOOKAHEAD_OFFSET_1_M, OFFSET_CAP_M),
    (IDX_LOOKAHEAD_OFFSET_2_M, OFFSET_CAP_M),
)
_NONNEGATIVE_CLIP_CAPS: tuple[tuple[int, float], ...] = (
    (IDX_WALL_FRONT_M, LIDAR_CAP_M),
    (IDX_WALL_FRONT_LEFT_M, LIDAR_CAP_M),
    (IDX_WALL_FRONT_RIGHT_M, LIDAR_CAP_M),
    (IDX_WALL_LEFT_M, LIDAR_CAP_M),
    (IDX_WALL_RIGHT_M, LIDAR_CAP_M),
)


def clip_state(state: np.ndarray) -> np.ndarray:
    """Clip a batched predicted state back to physically-sane bounds.

    A multi-step MPC rollout feeds the model's own output back in as its next
    input; an unlikely candidate action sequence can push a learned model to
    extrapolate into unphysical territory (which can also mean denormalized
    floats that are dramatically slower to compute with, not just a bad
    prediction). Called once per rollout step, after every
    `predict_next_state`, before the state is used for reward or fed back in.
    """
    clipped = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
    for index, cap in _SYMMETRIC_CLIP_CAPS:
        clipped[:, index] = np.clip(clipped[:, index], -cap, cap)
    for index, cap in _NONNEGATIVE_CLIP_CAPS:
        clipped[:, index] = np.clip(clipped[:, index], 0.0, cap)
    clipped[:, IDX_HEADING_ERROR_DEG] = np.clip(clipped[:, IDX_HEADING_ERROR_DEG], -180.0, 180.0)
    clipped[:, IDX_WALL_CONTACT] = np.clip(clipped[:, IDX_WALL_CONTACT], 0.0, 1.0)
    return clipped


def command_to_action(command: RobotCommand) -> np.ndarray:
    """Encode a `RobotCommand` as a float32 action vector."""
    return np.asarray((command.throttle, command.steer), dtype=np.float32)


def action_to_command(action: np.ndarray) -> RobotCommand:
    """Decode a float32 action vector back into a clamped `RobotCommand`."""
    throttle = float(np.clip(action[0], -1.0, 1.0))
    steer = float(np.clip(action[1], -1.0, 1.0))
    return RobotCommand(throttle=throttle, steer=steer)


def _clip(value: float, cap: float) -> float:
    return float(np.clip(value, -cap, cap))


def _clip_lidar(distance_m: float) -> float:
    if not np.isfinite(distance_m):
        return LIDAR_CAP_M
    return float(np.clip(distance_m, 0.0, LIDAR_CAP_M))
