"""Grid-search scaling of V0's learned throttle and steering residuals."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_controller import summarize
from train_v0 import SoloRaceRunner

from controllers.v0 import Controller, load_policy_parameters, save_policy_parameters


def _values(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(","))
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("scales must be positive comma-separated numbers")
    return values


def _seeds(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split(","))
    if not values:
        raise argparse.ArgumentTypeError("provide at least one seed")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throttle-scales", type=_values, default=(0.8, 1.0, 1.2))
    parser.add_argument("--steer-scales", type=_values, default=(0.8, 1.0, 1.2))
    parser.add_argument("--seeds", type=_seeds, default=(110, 2026, 500, 1337, 7777))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("src/controllers/v5_weights.json"))
    args = parser.parse_args()

    original = load_policy_parameters()
    best_score = float("-inf")
    best_parameters = original
    best_scales = (1.0, 1.0)
    runner = SoloRaceRunner()
    try:
        for throttle_scale in args.throttle_scales:
            for steer_scale in args.steer_scales:
                parameters = original.copy()
                parameters.actor_weights[0] = [value * throttle_scale for value in parameters.actor_weights[0]]
                parameters.actor_weights[1] = [value * steer_scale for value in parameters.actor_weights[1]]
                results = [
                    runner.run(Controller(parameters.copy()), seed=seed, duration_seconds=args.seconds)
                    for seed in args.seeds
                ]
                summary = summarize(results)
                print(
                    f"throttle={throttle_scale:.2f} steer={steer_scale:.2f}: "
                    f"score={summary.score:.2f}, lap={summary.best_lap_seconds}, "
                    f"min={summary.minimum_progress_m:.1f}, wall={summary.total_wall_contact_seconds:.2f}"
                )
                if summary.score > best_score:
                    best_score = summary.score
                    best_parameters = parameters
                    best_scales = (throttle_scale, steer_scale)
    finally:
        runner.close()

    best_parameters.best_evaluation_score = best_score
    save_policy_parameters(best_parameters, args.output)
    print(f"saved throttle={best_scales[0]:.2f}, steer={best_scales[1]:.2f}, score={best_score:.2f} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
