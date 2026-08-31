"""Turn vs straightaway classification from sensor-only signals.

`sensors.camera.lookahead_offsets_m` reports lateral offsets to points ahead
on the track centerline, in the car's current left/right frame. For a
constant-curvature arc and small angles, offset(d) ~= 0.5 * curvature(rad/m)
* d^2, so the farthest lookahead point gives a usable curvature estimate
without any privileged track state -- everything here comes from the public
`RobotSensors` contract, so it works both for the runtime controller and for
offline analysis scripts.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from racing import RobotSensors

Segment = Literal["turn", "straight"]

FAR_LOOKAHEAD_DISTANCE_M = 16.0  # matches SENSORS.md default lookahead_distances_m[-1]

# Calibrated from scripts/analyze_segment_curvature.py telemetry of a
# controller that already stays on-track: |curvature| is ~0.7 deg/m at the
# median (gentle drift/wander) and climbs past ~2 deg/m (turn radius under
# ~30m) through visible corners. The thresholds sit around the 65th-80th
# percentile of that live telemetry, past typical straightaway noise.
TURN_CURVATURE_THRESHOLD_DEG_PER_M = 2.0
TURN_HEADING_ERROR_THRESHOLD_DEG = 6.0


def curvature_degrees_per_m_from_offset(
    far_lookahead_offset_m: np.ndarray, *, distance_m: float = FAR_LOOKAHEAD_DISTANCE_M
) -> np.ndarray:
    """Shared small-angle curvature estimate: offset(d) ~= 0.5 * curvature(rad/m) * d^2.

    Takes a batch of predicted rollout offsets, so the MPC planner can blend
    its reward toward the turn profile at every imagined step. The live
    sensor-based classifier below uses the same formula on a single reading.
    """
    curvature_rad_per_m = 2.0 * far_lookahead_offset_m / (distance_m**2)
    return np.degrees(curvature_rad_per_m)


def estimate_curvature_degrees_per_m(sensors: RobotSensors) -> float:
    """Estimate signed track curvature ahead, in degrees turned per meter."""
    offsets = sensors.camera.lookahead_offsets_m
    distances = sensors.camera.lookahead_distances_m
    if len(offsets) == 0 or len(distances) == 0:
        return 0.0
    distance = distances[-1]
    if distance <= 0.0:
        return 0.0
    curvature_rad_per_m = 2.0 * offsets[-1] / (distance**2)
    return math.degrees(curvature_rad_per_m)


def turn_blend_weight(curvature_deg_per_m: np.ndarray) -> np.ndarray:
    """Return a continuous 0 (straight) to 1 (turn) blend weight from curvature magnitude."""
    return np.clip(np.abs(curvature_deg_per_m) / TURN_CURVATURE_THRESHOLD_DEG_PER_M, 0.0, 1.0)


def classify_segment(sensors: RobotSensors) -> Segment:
    """Classify the upcoming track as a turn or a straightaway from sensors alone."""
    curvature_deg_per_m = abs(estimate_curvature_degrees_per_m(sensors))
    heading_error_deg = abs(sensors.camera.heading_error_degrees)
    curved_ahead = curvature_deg_per_m >= TURN_CURVATURE_THRESHOLD_DEG_PER_M
    misaligned = heading_error_deg >= TURN_HEADING_ERROR_THRESHOLD_DEG
    if curved_ahead or misaligned:
        return "turn"
    return "straight"
