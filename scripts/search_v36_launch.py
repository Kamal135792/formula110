"""Search the stable-launch duration for the partition controller."""

from __future__ import annotations

import argparse

from benchmark_controller import summarize
from controllers.v36 import Controller
from train_v0 import SoloRaceRunner


def _integers(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=_integers, default=(0, 60, 120, 180, 240, 300))
    parser.add_argument("--seeds", type=_integers, default=(110, 2026, 42, 73, 500, 901, 1337, 4096, 7777, 9999))
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    best = (float("-inf"), 0)
    runner = SoloRaceRunner()
    try:
        for ticks in args.ticks:
            results = [
                runner.run(
                    Controller(boost_enabled=False, stable_launch_ticks=ticks),
                    seed=seed,
                    duration_seconds=args.seconds,
                )
                for seed in args.seeds
            ]
            summary = summarize(results)
            print(
                f"ticks={ticks} ({ticks / 60:.1f}s): score={summary.score:.2f}, "
                f"progress={summary.mean_progress_m:.1f}, lap={summary.best_lap_seconds}, "
                f"min={summary.minimum_progress_m:.1f}, wall={summary.total_wall_contact_seconds:.2f}"
            )
            if summary.score > best[0]:
                best = (summary.score, ticks)
    finally:
        runner.close()
    print(f"best score={best[0]:.2f}, ticks={best[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
