"""Per-tick reward for training the hybrid controller's RL correction residual.

Phase 3 of the improvement plan (`controllers/exploration_faster.py`): the
deterministic racing line already handles basic track-following, so this
reward is not "stay near the centerline" in isolation -- it is "make forward
progress, stay fast, stay on the line, and above all stay off the walls."
`DAMAGE_PENALTY_WEIGHT` is deliberately large relative to the other terms:
the first hand-tuned residual experiment traded wall contact for lap time,
so this reward makes that trade unprofitable by construction.
"""

from __future__ import annotations

from math import isfinite

from racing import RobotCommand, RobotSensors

PROGRESS_WEIGHT = 1.0
SPEED_BONUS_WEIGHT = 0.05
LINE_DEVIATION_WEIGHT = 0.6
DAMAGE_PENALTY_WEIGHT = 60.0
WALL_CONTACT_PENALTY_WEIGHT = 3.0
WALL_PROXIMITY_WEIGHT = 1.0
WALL_PROXIMITY_MARGIN_M = 2.0
UNNECESSARY_BRAKING_WEIGHT = 0.2
STEER_JERK_WEIGHT = 0.05

# A curve/tracking-error/front-hazard state below these thresholds means
# braking has no obvious justification, so throttle < 0 there is penalized
# as "unnecessary braking" per the plan's reward structure.
BRAKING_CURVE_SEVERITY_THRESHOLD = 0.15
BRAKING_TRACKING_ERROR_THRESHOLD = 0.25
BRAKING_FRONT_CLEAR_M = 8.0


def _curve_severity(sensors: RobotSensors) -> float:
    offsets = sensors.camera.lookahead_offsets_m
    if len(offsets) < 3:
        return abs(sensors.camera.heading_error_degrees) / 55.0
    return max(
        abs(sensors.camera.heading_error_degrees) / 55.0,
        0.62 * abs(offsets[0]) / 4.0,
        0.90 * abs(offsets[1]) / 9.0,
        1.25 * abs(offsets[2]) / 16.0,
    )


def step_reward(
    *,
    sensors: RobotSensors,
    command: RobotCommand,
    previous_command: RobotCommand,
    progress_delta_m: float,
    previous_damage: float,
) -> float:
    """Reward for the tick that produced `command` and landed in `sensors`/`progress_delta_m`."""
    progress = PROGRESS_WEIGHT * max(0.0, progress_delta_m)
    speed_bonus = SPEED_BONUS_WEIGHT * max(0.0, sensors.odometry.speed_mps)

    line_deviation = LINE_DEVIATION_WEIGHT * (sensors.camera.center_offset_m / 2.0) ** 2

    damage_delta = max(0.0, sensors.contact.damage - previous_damage)
    damage_penalty = DAMAGE_PENALTY_WEIGHT * damage_delta
    wall_contact_penalty = WALL_CONTACT_PENALTY_WEIGHT * sensors.dt_s if sensors.contact.wall > 0.0 else 0.0

    left_m = sensors.wall_lidar.left_m if isfinite(sensors.wall_lidar.left_m) else WALL_PROXIMITY_MARGIN_M
    right_m = sensors.wall_lidar.right_m if isfinite(sensors.wall_lidar.right_m) else WALL_PROXIMITY_MARGIN_M
    front_m = sensors.wall_lidar.front_m if isfinite(sensors.wall_lidar.front_m) else WALL_PROXIMITY_MARGIN_M
    nearest_wall_m = max(0.0, min(left_m, right_m, front_m))
    wall_proximity_penalty = WALL_PROXIMITY_WEIGHT * max(0.0, WALL_PROXIMITY_MARGIN_M - nearest_wall_m) ** 2

    curve_severity = _curve_severity(sensors)
    tracking_error = max(
        abs(sensors.camera.center_offset_m) / 2.4,
        abs(sensors.camera.heading_error_degrees) / 55.0,
    )
    braking_looks_unnecessary = (
        command.throttle < 0.0
        and curve_severity < BRAKING_CURVE_SEVERITY_THRESHOLD
        and tracking_error < BRAKING_TRACKING_ERROR_THRESHOLD
        and front_m > BRAKING_FRONT_CLEAR_M
        and sensors.contact.wall <= 0.0
    )
    unnecessary_braking_penalty = UNNECESSARY_BRAKING_WEIGHT * (-command.throttle) if braking_looks_unnecessary else 0.0

    steer_jerk_penalty = STEER_JERK_WEIGHT * (command.steer - previous_command.steer) ** 2

    return (
        progress
        + speed_bonus
        - line_deviation
        - damage_penalty
        - wall_contact_penalty
        - wall_proximity_penalty
        - unnecessary_braking_penalty
        - steer_jerk_penalty
    )
