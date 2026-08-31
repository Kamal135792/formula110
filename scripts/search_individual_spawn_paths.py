"""Find distinct zero-damage first-corner paths for difficult spawn classes."""

from __future__ import annotations

import argparse
from random import Random

from benchmark_controller import _parse_seeds
from train_v0 import SoloRaceRunner

from controllers.v64 import Controller


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=(64, 68, 78, 86, 93, 97))
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--minimum-progress", type=float, default=60.0)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    random = Random(64068)
    runner = SoloRaceRunner()
    try:
        for seed in args.seeds:
            best: tuple[float, float, tuple[float, float, float, float, float]] | None = None
            for trial in range(args.trials):
                if trial == 0:
                    parameters = (35.0, 41.0, 48.0, 48.0, -0.20)
                elif trial == 1:
                    parameters = (38.0, 44.0, 48.0, 48.0, -0.175)
                else:
                    start = random.uniform(20.0, 41.5)
                    peak_start = random.uniform(max(start, 34.0), 47.0)
                    peak_end = random.uniform(max(peak_start, 44.0), 53.0)
                    end = random.uniform(max(peak_end, 48.0), 58.0)
                    correction = random.uniform(-0.42, 0.12)
                    parameters = (start, peak_start, peak_end, end, correction)
                result = runner.run(
                    Controller(
                        guard_start_m=parameters[0],
                        guard_peak_start_m=parameters[1],
                        guard_peak_end_m=parameters[2],
                        guard_end_m=parameters[3],
                        avoidance_steer=parameters[4],
                        repeat_guard=args.repeat,
                    ),
                    seed=seed,
                    duration_seconds=args.seconds,
                )
                rank = (result.damage, -result.raw_progress_m, parameters)
                if best is None or rank < best:
                    best = rank
                    print(
                        f"seed={seed} trial={trial:>2} damage={result.damage:.8f} "
                        f"progress={result.raw_progress_m:.2f} path={tuple(round(v, 4) for v in parameters)}"
                    )
                if result.damage == 0.0 and result.raw_progress_m >= args.minimum_progress:
                    break
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
