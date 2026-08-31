"""Search phase-aware racing-line offsets for controller V29."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any, cast

from benchmark_controller import summarize
from train_v0 import SoloRaceRunner


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def _integers(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="controllers.v29")
    parser.add_argument("--outside", type=_floats, default=(0.0, 0.2, 0.4, 0.6))
    parser.add_argument("--inside", type=_floats, default=(0.0, 0.2, 0.4, 0.6))
    parser.add_argument("--gain", type=float, default=0.06)
    parser.add_argument("--seeds", type=_integers, default=(18, 29, 110, 2026, 7777))
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    module = cast(Any, import_module(args.module))
    best = (float("-inf"), 0.0, 0.0)
    runner = SoloRaceRunner()
    try:
        for outside in args.outside:
            for inside in args.inside:
                if hasattr(module, "OUTSIDE_OFFSET_M"):
                    module.OUTSIDE_OFFSET_M = outside
                module.INSIDE_OFFSET_M = inside
                module.LINE_STEER_GAIN = args.gain
                results = [
                    runner.run(module.Controller(), seed=seed, duration_seconds=args.seconds) for seed in args.seeds
                ]
                summary = summarize(results)
                print(
                    f"outside={outside:.2f} inside={inside:.2f}: score={summary.score:.2f}, "
                    f"progress={summary.mean_progress_m:.1f}, lap={summary.best_lap_seconds}, "
                    f"min={summary.minimum_progress_m:.1f}, wall={summary.total_wall_contact_seconds:.2f}"
                )
                if summary.score > best[0]:
                    best = (summary.score, outside, inside)
    finally:
        runner.close()
    print(f"best score={best[0]:.2f}, outside={best[1]:.2f}, inside={best[2]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
