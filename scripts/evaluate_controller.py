"""Phase 1 & 6 of the hybrid-controller improvement plan: baseline and regression evaluation.

Runs a controller (default: `controllers.exploration_faster`) across many
seeds and races, and reports the metrics the plan asks for: lap time,
completion rate, damage/wall contact, average speed, time braking vs
accelerating, distance covered, and a per-sector time breakdown (to find the
slowest/least consistent section -- Phase 5's "identify bottleneck" step).

Every seed races the controller against a second copy of itself, so both
"sides" of the head-to-head are the controller under test and both are
recorded -- this doubles the effective sample count per seed instead of
wasting half the race on an unused opponent.

Usage:
    uv run python scripts/evaluate_controller.py
    uv run python scripts/evaluate_controller.py --seeds 42 110 271 997 2027 7 13 21 55 89
    uv run python scripts/evaluate_controller.py --tuning-path artifacts/hybrid_tuning_candidate.json \\
        --report artifacts/eval_candidate.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from controllers.dynamics.hybrid_telemetry import Episode

DEFAULT_SEEDS: tuple[int, ...] = (42, 110, 271, 997, 2027, 7, 13, 21, 55, 89)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the hybrid controller across seeds (Phase 1 / 6).")
    parser.add_argument("--module", default="controllers.exploration_faster")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--races-per-seed", type=int, default=1)
    parser.add_argument("--round-seconds", type=float, default=30.0)
    parser.add_argument("--tuning-path", type=Path, default=None, help="hybrid_tuning.json override to evaluate")
    parser.add_argument("--report", type=Path, default=None, help="write the full JSON report here")
    return parser


def evaluate(*, module_path: str, seeds: list[int], races_per_seed: int, round_seconds: float) -> dict[str, object]:
    # Imported here, not at module scope: `HYBRID_TUNING_PATH` must already be
    # set in the environment (see `main`) before `controllers.exploration_faster`
    # is first imported, since it reads that env var once at import time.
    from controllers.dynamics.hybrid_telemetry import TelemetryController
    from racing import run_headless_head_to_head
    from racing.race.rules import HeadToHeadRaceRules

    target = import_module(module_path)

    all_episodes: list[Episode] = []
    best_lap_times_s: list[float] = []
    completed_lap_flags: list[bool] = []
    damages: list[float] = []
    wall_contact_seconds: list[float] = []
    distances_m: list[float] = []

    for seed in seeds:
        episodes: list[Episode] = []
        challenger = TelemetryController(target.create_controller, episodes=episodes)
        incumbent = TelemetryController(target.create_controller, episodes=episodes)
        result = run_headless_head_to_head(
            challenger_controller=challenger,
            incumbent_controller=incumbent,
            race_count=races_per_seed,
            round_seconds=round_seconds,
            random_seed=seed,
            rules=HeadToHeadRaceRules(marshal_enabled=False),
        )
        for race in result.races:
            for stats in (race.challenger, race.incumbent):
                best_lap = stats.best_lap_times_seconds[0]
                if best_lap is not None:
                    best_lap_times_s.append(best_lap)
                completed_lap_flags.append(stats.lap_counts[0] > 0)
                damages.append(stats.damages[0])
                wall_contact_seconds.append(stats.wall_contact_seconds[0])
                distances_m.append(stats.distances_m[0])
        all_episodes.extend(episodes)
        print(f"seed {seed}: {len(episodes)} episodes recorded")

    return _summarize(all_episodes, best_lap_times_s, completed_lap_flags, damages, wall_contact_seconds, distances_m)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summarize(
    episodes: list[Episode],
    best_lap_times_s: list[float],
    completed_lap_flags: list[bool],
    damages: list[float],
    wall_contact_seconds: list[float],
    distances_m: list[float],
) -> dict[str, object]:
    from controllers.dynamics.hybrid_telemetry import sector_visit_durations

    avg_speeds: list[float] = []
    brake_fractions: list[float] = []
    accel_fractions: list[float] = []
    sector_durations: dict[int, list[float]] = {}

    for episode in episodes:
        if not episode.ticks:
            continue
        total_dt = sum(tick.dt_s for tick in episode.ticks)
        if total_dt <= 0.0:
            continue
        avg_speeds.append(statistics.fmean(tick.speed_mps for tick in episode.ticks))
        brake_time = sum(tick.dt_s for tick in episode.ticks if tick.throttle < -0.02)
        accel_time = sum(tick.dt_s for tick in episode.ticks if tick.throttle > 0.02)
        brake_fractions.append(brake_time / total_dt)
        accel_fractions.append(accel_time / total_dt)
        for sector, durations in sector_visit_durations(episode).items():
            sector_durations.setdefault(sector, []).extend(durations)

    return {
        "episode_count": len(episodes),
        "lap_time_s": {
            "best": min(best_lap_times_s) if best_lap_times_s else None,
            "mean": _mean(best_lap_times_s),
            "laps_completed": len(best_lap_times_s),
        },
        "completion_rate": (sum(completed_lap_flags) / len(completed_lap_flags)) if completed_lap_flags else None,
        "damage": {"mean": _mean(damages), "max": max(damages) if damages else None},
        "wall_contact_s": {
            "mean": _mean(wall_contact_seconds),
            "max": max(wall_contact_seconds) if wall_contact_seconds else None,
        },
        "distance_m": {"mean": _mean(distances_m), "min": min(distances_m) if distances_m else None},
        "avg_speed_mps": _mean(avg_speeds),
        "brake_time_fraction": _mean(brake_fractions),
        "accel_time_fraction": _mean(accel_fractions),
        "sector_time_s": {
            str(sector): {
                "mean": statistics.fmean(durations),
                "stdev": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
                "visits": len(durations),
            }
            for sector, durations in sorted(sector_durations.items())
        },
    }


def _print_report(report: dict[str, object]) -> None:
    lap = report["lap_time_s"]
    print("\n=== summary ===")
    print(f"episodes:          {report['episode_count']}")
    print(f"laps completed:    {lap['laps_completed']} (best {lap['best']}, mean {lap['mean']})")
    completion_rate = report["completion_rate"]
    print(f"completion rate:   {completion_rate:.1%}" if completion_rate is not None else "completion rate:   n/a")
    damage = report["damage"]
    print(f"damage:            mean {damage['mean']:.4f}, max {damage['max']:.4f}")
    wall = report["wall_contact_s"]
    print(f"wall contact (s):  mean {wall['mean']:.3f}, max {wall['max']:.3f}")
    dist = report["distance_m"]
    print(f"distance (m):      mean {dist['mean']:.1f}, min {dist['min']:.1f}")
    print(f"avg speed (m/s):   {report['avg_speed_mps']:.2f}")
    print(f"time braking:      {report['brake_time_fraction']:.1%}")
    print(f"time accelerating: {report['accel_time_fraction']:.1%}")
    print("\nsector time (s), sorted slowest first:")
    sector_rows = sorted(report["sector_time_s"].items(), key=lambda item: -item[1]["mean"])
    for sector, stats in sector_rows:
        print(f"  sector {sector}: mean {stats['mean']:.3f}s  stdev {stats['stdev']:.3f}  visits {stats['visits']}")


def main() -> None:
    args = build_parser().parse_args()
    if args.tuning_path is not None:
        os.environ["HYBRID_TUNING_PATH"] = str(args.tuning_path)

    report = evaluate(
        module_path=args.module,
        seeds=list(args.seeds),
        races_per_seed=int(args.races_per_seed),
        round_seconds=float(args.round_seconds),
    )
    _print_report(report)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
