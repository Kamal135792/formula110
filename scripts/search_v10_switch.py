"""Search the stable-to-sprint handoff time for controller V10."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any, cast

from benchmark_controller import summarize
from train_v0 import SoloRaceRunner


def _integers(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="controllers.v10")
    parser.add_argument("--ticks", type=_integers, default=(180, 300, 420, 600))
    parser.add_argument("--seeds", type=_integers, default=(110, 2026, 42, 73, 500, 901, 1337, 4096, 7777, 9999))
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    best_score = float("-inf")
    module = cast(Any, import_module(args.module))
    best_tick = int(module.SPRINT_START_TICK)
    runner = SoloRaceRunner()
    try:
        for tick in args.ticks:
            module.SPRINT_START_TICK = tick
            results = [runner.run(module.Controller(), seed=seed, duration_seconds=args.seconds) for seed in args.seeds]
            summary = summarize(results)
            print(
                f"tick={tick} ({tick / 60:.1f}s): score={summary.score:.2f}, "
                f"progress={summary.mean_progress_m:.1f}, min={summary.minimum_progress_m:.1f}, "
                f"lap={summary.best_lap_seconds}, wall={summary.total_wall_contact_seconds:.2f}"
            )
            if summary.score > best_score:
                best_score = summary.score
                best_tick = tick
    finally:
        runner.close()
    print(f"best tick={best_tick} ({best_tick / 60:.1f}s), score={best_score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
