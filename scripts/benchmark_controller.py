"""Benchmark any student controller in repeatable headless solo races."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import cast

from train_v0 import EvaluationSummary, SoloEpisodeResult, SoloRaceRunner

from racing import load_student_submission
from racing.student.api import RobotController


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _fresh_controller(controller: RobotController) -> RobotController:
    copy_for_car = getattr(controller, "copy_for_car", None)
    return cast(RobotController, copy_for_car()) if callable(copy_for_car) else controller


def summarize(results: list[SoloEpisodeResult]) -> EvaluationSummary:
    lap_times = [result.best_lap_seconds for result in results if result.best_lap_seconds is not None]
    return EvaluationSummary(
        score=mean(result.evaluation_score for result in results),
        mean_progress_m=mean(result.raw_progress_m for result in results),
        minimum_progress_m=min(result.raw_progress_m for result in results),
        total_laps=sum(result.lap_count for result in results),
        best_lap_seconds=min(lap_times) if lap_times else None,
        mean_max_speed_mps=mean(result.max_speed_mps for result in results),
        mean_damage=mean(result.damage for result in results),
        total_wall_contact_seconds=sum(result.wall_contact_seconds for result in results),
        total_off_track_seconds=sum(result.off_track_seconds for result in results),
        elimination_count=sum(result.eliminated for result in results),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="controller module, for example controllers.v1")
    parser.add_argument("--seeds", type=_parse_seeds, default=(110, 2026))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    submission = load_student_submission(args.module)
    runner = SoloRaceRunner()
    try:
        results = [
            runner.run(_fresh_controller(submission.controller), seed=seed, duration_seconds=args.seconds)
            for seed in args.seeds
        ]
    finally:
        runner.close()
    summary = summarize(results)
    lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.2f}s"
    print(
        f"{args.module}: score={summary.score:.2f}, progress={summary.mean_progress_m:.1f}m "
        f"(min {summary.minimum_progress_m:.1f}m), laps={summary.total_laps}, best_lap={lap}, "
        f"speed={summary.mean_max_speed_mps:.2f}m/s, damage={summary.mean_damage:.3f}, "
        f"wall={summary.total_wall_contact_seconds:.2f}s, offtrack={summary.total_off_track_seconds:.2f}s"
    )
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"summary": asdict(summary), "episodes": [asdict(result) for result in results]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
