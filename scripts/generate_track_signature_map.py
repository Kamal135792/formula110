"""Generate a controller-local track signature map from public camera geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from racing.race.progress import default_track_progress_model, track_pose_at_distance
from racing.race.sensors import camera_sensors_from_track


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("src/controllers/track_signature_map.json"))
    args = parser.parse_args()
    if args.spacing <= 0.0:
        raise ValueError("spacing must be positive")

    model = default_track_progress_model()
    sample_count = round(model.total_length_m / args.spacing)
    records: list[list[float]] = []
    for index in range(sample_count):
        progress = index * model.total_length_m / sample_count
        pose = track_pose_at_distance(model, progress)
        camera = camera_sensors_from_track(
            model=model,
            position=pose.position,
            heading_degrees=pose.heading_degrees,
        )
        records.append(
            [
                round(progress, 6),
                round(pose.heading_degrees, 6),
                *(round(value, 6) for value in camera.lookahead_offsets_m),
            ]
        )

    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_length_m": model.total_length_m,
                "spacing_m": model.total_length_m / sample_count,
                "samples": records,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {sample_count} signatures for {model.total_length_m:.2f} m track to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
