"""Grid-search V0 geometric planner parameters against fixed seeds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmark_controller import summarize
from train_v0 import SoloRaceRunner

from controllers.v0 import Controller, PlannerConfig, load_policy_parameters


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def _seeds(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=_floats, default=(31.0, 33.0, 35.0))
    parser.add_argument("--curve-gains", type=_floats, default=(1.0, 1.15, 1.3))
    parser.add_argument("--steer-gains", type=_floats, default=(1.65,))
    parser.add_argument("--apex-biases", type=_floats, default=(0.0,))
    parser.add_argument("--seeds", type=_seeds, default=(110, 2026, 500, 1337, 7777))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("src/controllers/v6_config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("src/controllers/v0_weights.json"))
    args = parser.parse_args()

    parameters = load_policy_parameters(args.checkpoint)
    best_score = float("-inf")
    best_config = PlannerConfig()
    runner = SoloRaceRunner()
    try:
        for speed in args.speeds:
            for curve_gain in args.curve_gains:
                for steer_gain in args.steer_gains:
                    for apex_bias in args.apex_biases:
                        config = PlannerConfig(
                            pure_pursuit_gain=steer_gain,
                            speed_numerator=speed,
                            curve_speed_gain=curve_gain,
                            minimum_corner_speed=min(12.0, speed),
                            apex_bias_m=apex_bias,
                        )
                        results = [
                            runner.run(
                                Controller(parameters.copy(), planner_config=config),
                                seed=seed,
                                duration_seconds=args.seconds,
                            )
                            for seed in args.seeds
                        ]
                        summary = summarize(results)
                        print(
                            f"speed={speed:.1f} curve={curve_gain:.2f} steer={steer_gain:.2f} "
                            f"apex={apex_bias:.2f}: score={summary.score:.2f}, "
                            f"lap={summary.best_lap_seconds}, min={summary.minimum_progress_m:.1f}, "
                            f"wall={summary.total_wall_contact_seconds:.2f}"
                        )
                        if summary.score > best_score:
                            best_score = summary.score
                            best_config = config
    finally:
        runner.close()

    args.output.write_text(
        json.dumps({"score": best_score, "planner": asdict(best_config)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved score={best_score:.2f}, config={best_config} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
