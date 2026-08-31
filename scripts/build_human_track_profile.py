"""Convert recorded manual driving into a localized per-metre track profile."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import floor
from pathlib import Path
from statistics import mean
from typing import cast

from controllers.track_localizer import TrackLocalizer


@dataclass(slots=True)
class BinSamples:
    speeds: list[float]
    center_offsets: list[float]
    heading_errors: list[float]
    throttles: list[float]
    steers: list[float]


def _number(record: dict[str, object], key: str) -> float:
    value = record[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _numeric_tuple(record: dict[str, object], key: str) -> tuple[float, ...]:
    values = record[key]
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a list")
    return tuple(float(value) for value in cast(list[float], values))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _new_bins(count: int) -> list[BinSamples]:
    return [BinSamples([], [], [], [], []) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, nargs="+", help="one or more --record-human JSONL files")
    parser.add_argument("--output", type=Path, default=Path("artifacts/human_track_profile.json"))
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--minimum-samples", type=int, default=5)
    args = parser.parse_args()
    if args.bin_width <= 0.0 or args.minimum_samples < 1:
        raise ValueError("bin width and minimum samples must be positive")

    localizer = TrackLocalizer()
    bin_count = int(localizer.total_length_m / args.bin_width) + 1
    bins = _new_bins(bin_count)
    sessions: dict[str, tuple[float, float]] = {}
    accepted_samples = 0
    rejected_contact_samples = 0

    for path in cast(list[Path], args.recordings):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            unchecked: object = json.loads(line)
            if not isinstance(unchecked, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            record = cast(dict[str, object], unchecked)
            if record.get("schema_version") != 2 or record.get("record_type") != "human_control_step":
                raise ValueError(f"{path}:{line_number}: unsupported human recording row")
            session_id = str(record["session_id"])
            sensors = cast(dict[str, object], record["sensors"])
            imu = cast(dict[str, object], sensors["imu"])
            odometry = cast(dict[str, object], sensors["odometry"])
            camera = cast(dict[str, object], sensors["camera"])
            contact = cast(dict[str, object], sensors["contact"])
            command = cast(dict[str, object], record["command"])

            distance_m = _number(odometry, "distance_m")
            if session_id not in sessions:
                desired_heading = _number(imu, "heading_degrees") + _number(camera, "heading_error_degrees")
                initial_progress = localizer.locate_signature(
                    desired_heading_degrees=desired_heading,
                    center_offset_m=_number(camera, "center_offset_m"),
                    lookahead_offsets_m=_numeric_tuple(camera, "lookahead_offsets_m"),
                )
                sessions[session_id] = (initial_progress, distance_m)
            initial_progress, initial_distance = sessions[session_id]
            progress = (initial_progress + max(0.0, distance_m - initial_distance)) % localizer.total_length_m

            if _number(contact, "wall") > 0.0 or _number(contact, "damage") > 0.98:
                rejected_contact_samples += 1
                continue
            if not bool(camera["visible"]):
                continue
            bin_index = min(bin_count - 1, floor(progress / args.bin_width))
            samples = bins[bin_index]
            samples.speeds.append(_number(odometry, "speed_mps"))
            samples.center_offsets.append(_number(camera, "center_offset_m"))
            samples.heading_errors.append(_number(camera, "heading_error_degrees"))
            samples.throttles.append(_number(command, "throttle"))
            samples.steers.append(_number(command, "steer"))
            accepted_samples += 1

    profile_bins: list[dict[str, float | int]] = []
    estimated_lap_seconds = 0.0
    covered_distance_m = 0.0
    for index, samples in enumerate(bins):
        if len(samples.speeds) < args.minimum_samples:
            continue
        mean_speed = mean(samples.speeds)
        profile_bins.append(
            {
                "start_m": index * args.bin_width,
                "sample_count": len(samples.speeds),
                "mean_speed_mps": mean_speed,
                "p90_speed_mps": _percentile(samples.speeds, 0.90),
                "mean_center_offset_m": mean(samples.center_offsets),
                "mean_heading_error_degrees": mean(samples.heading_errors),
                "mean_throttle": mean(samples.throttles),
                "mean_steer": mean(samples.steers),
            }
        )
        if mean_speed > 0.5:
            estimated_lap_seconds += args.bin_width / mean_speed
            covered_distance_m += args.bin_width

    output = cast(Path, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "track_length_m": localizer.total_length_m,
                "bin_width_m": args.bin_width,
                "session_count": len(sessions),
                "accepted_samples": accepted_samples,
                "rejected_contact_samples": rejected_contact_samples,
                "covered_distance_m": covered_distance_m,
                "estimated_lap_seconds_over_covered_bins": estimated_lap_seconds,
                "bins": profile_bins,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output} with {len(profile_bins)}/{bin_count} bins from {len(sessions)} sessions "
        f"({accepted_samples} accepted samples, {rejected_contact_samples} contact samples removed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
