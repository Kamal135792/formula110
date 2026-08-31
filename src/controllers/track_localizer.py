"""Lightweight localization against a fixed map of public camera signatures."""

from __future__ import annotations

import json
from math import fmod
from pathlib import Path
from typing import cast

from racing import RobotSensors

_MAP_PATH = Path(__file__).with_name("track_signature_map.json")


def _angle_error_degrees(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def _wrapped_distance(first: float, second: float, total: float) -> float:
    return abs((first - second + total / 2.0) % total - total / 2.0)


class TrackLocalizer:
    """Estimate centerline progress using heading, camera shape, and odometry."""

    def __init__(self) -> None:
        record = cast(dict[str, object], json.loads(_MAP_PATH.read_text(encoding="utf-8")))
        self.total_length_m = float(cast(float, record["total_length_m"]))
        rows = cast(list[list[float]], record["samples"])
        self._samples = tuple(tuple(float(value) for value in row) for row in rows)
        self.progress_m: float | None = None
        self._last_odometry_m = 0.0

    def update(self, sensors: RobotSensors) -> float:
        if sensors.tick == 0:
            self.progress_m = None
            self._last_odometry_m = sensors.odometry.distance_m

        odometry_delta = max(0.0, sensors.odometry.distance_m - self._last_odometry_m)
        self._last_odometry_m = sensors.odometry.distance_m
        predicted = None if self.progress_m is None else (self.progress_m + odometry_delta) % self.total_length_m

        desired_heading = sensors.imu.heading_degrees + sensors.camera.heading_error_degrees
        best_progress = self.locate_signature(
            desired_heading_degrees=desired_heading,
            center_offset_m=sensors.camera.center_offset_m,
            lookahead_offsets_m=sensors.camera.lookahead_offsets_m,
            predicted_progress_m=predicted,
        )
        self.progress_m = fmod(best_progress, self.total_length_m)
        return self.progress_m

    def locate_signature(
        self,
        *,
        desired_heading_degrees: float,
        center_offset_m: float,
        lookahead_offsets_m: tuple[float, ...],
        predicted_progress_m: float | None = None,
    ) -> float:
        """Match a public camera/heading signature to centerline progress."""
        offsets = tuple(value - center_offset_m for value in lookahead_offsets_m)
        padded = (*offsets, 0.0, 0.0, 0.0)

        best_progress = 0.0
        best_error = float("inf")
        for row in self._samples:
            progress, heading, near, middle, far = row
            if (
                predicted_progress_m is not None
                and _wrapped_distance(progress, predicted_progress_m, self.total_length_m) > 10.0
            ):
                continue
            error = (
                (_angle_error_degrees(desired_heading_degrees, heading) / 10.0) ** 2
                + ((padded[0] - near) / 1.5) ** 2
                + ((padded[1] - middle) / 3.5) ** 2
                + ((padded[2] - far) / 6.0) ** 2
            )
            if predicted_progress_m is not None:
                error += (_wrapped_distance(progress, predicted_progress_m, self.total_length_m) / 4.0) ** 2
            if error < best_error:
                best_error = error
                best_progress = progress
        return best_progress
