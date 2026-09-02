"""Phases 2-5 of the hybrid-controller improvement plan: train `_TuningParams` against real races.

The physics simulator is a black box (no gradients), so this is a simple
(mu/mu, lambda) evolution strategy: sample a population of candidate
`_TuningParams` around the current best, evaluate each with a short headless
race using the Phase 3 reward (`controllers/dynamics/hybrid_reward.py`,
heavily penalizing damage per the plan), keep the best-scoring fraction,
move the search mean to their average, and repeat. `--focus-sector` reweights
the reward toward ticks spent in one sector (Phase 5: concentrate training on
the identified bottleneck rather than the whole track uniformly).

`--params` selects what to search over:
  residual  -- actor_weights, action_residual_scales, hazard_residual_scale_max
               (Phase 3/4: the RL correction policy and when it takes over)
  line      -- corner offsets, curve-speed gains, straight speeds
               (Phase 2/5: the racing line and speed schedule)
  all       -- both (default)

This only ever proposes a candidate; nothing is kept automatically. Use
scripts/evaluate_controller.py on the written --output file and compare
against a baseline run before adopting it (Phase 6: keep only if it actually
improves average performance without materially increasing damage).

Usage:
    uv run python scripts/train_hybrid_residual.py --generations 8 --population 16 \\
        --output artifacts/hybrid_tuning_trained.json
    uv run python scripts/train_hybrid_residual.py --focus-sector 1 --resume artifacts/hybrid_tuning_trained.json \\
        --output artifacts/hybrid_tuning_sector1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np

ParamGroup = Literal["residual", "line", "all"]

DEFAULT_SEEDS: tuple[int, ...] = (42, 110, 271, 997, 2027)

# Per-field search noise, roughly 10-25% of each field's natural scale (the
# baseline actor weights are O(0.005-0.01); residual scales and gains are
# O(0.3-1.5); speeds are O(35-45)). CMA-style adaptation is overkill for ~40
# parameters and a noisy few-seed fitness -- a fixed, group-appropriate sigma
# with elite averaging is simpler and stable enough for this budget.
ACTOR_WEIGHT_SIGMA = 0.0015
RESIDUAL_SCALE_SIGMA = 0.05
HAZARD_SCALE_SIGMA = 0.3
CORNER_OFFSET_SIGMA = 0.25
CURVE_GAIN_SIGMA = 0.08
STRAIGHT_SPEED_SIGMA = 1.5

# Keep the search in a physically sane neighborhood of the hand-tuned
# baseline so a bad generation cannot wander into a degenerate controller.
HAZARD_SCALE_BOUNDS = (1.0, 4.0)
RESIDUAL_SCALE_BOUNDS = (0.0, 0.8)
CORNER_OFFSET_BOUNDS = (0.0, 4.0)
CURVE_GAIN_BOUNDS = (0.1, 2.0)
STRAIGHT_SPEED_BOUNDS = (25.0, 55.0)

# A candidate whose mean damage exceeds the baseline's by more than this
# (absolute, 0..1 scale) is rejected outright regardless of reward -- Phase 6
# / Phase 3's "damage should receive a much larger penalty" applied at the
# selection level too, not just inside the reward.
MAX_DAMAGE_REGRESSION = 0.03


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return float(np.clip(value, bounds[0], bounds[1]))


def flatten(tuning: object, group: ParamGroup) -> np.ndarray:
    parts: list[float] = []
    if group in ("residual", "all"):
        parts.extend(tuning.actor_weights[0])
        parts.extend(tuning.actor_weights[1])
        parts.extend(tuning.action_residual_scales)
        parts.append(tuning.hazard_residual_scale_max)
    if group in ("line", "all"):
        parts.extend(tuning.launch_corner_offsets_m)
        parts.extend(tuning.launch_curve_speed_gains)
        parts.append(tuning.launch_straight_speed_mps)
        parts.extend(tuning.recurring_corner_offsets_m)
        parts.extend(tuning.recurring_curve_speed_gains)
        parts.append(tuning.recurring_straight_speed_mps)
    return np.asarray(parts, dtype=np.float64)


def sigma_vector(group: ParamGroup) -> np.ndarray:
    parts: list[float] = []
    if group in ("residual", "all"):
        parts.extend([ACTOR_WEIGHT_SIGMA] * 15)
        parts.extend([ACTOR_WEIGHT_SIGMA] * 15)
        parts.extend([RESIDUAL_SCALE_SIGMA] * 2)
        parts.append(HAZARD_SCALE_SIGMA)
    if group in ("line", "all"):
        parts.extend([CORNER_OFFSET_SIGMA] * 5)
        parts.extend([CURVE_GAIN_SIGMA] * 5)
        parts.append(STRAIGHT_SPEED_SIGMA)
        parts.extend([CORNER_OFFSET_SIGMA] * 5)
        parts.extend([CURVE_GAIN_SIGMA] * 5)
        parts.append(STRAIGHT_SPEED_SIGMA)
    return np.asarray(parts, dtype=np.float64)


def unflatten(vector: np.ndarray, base: object, group: ParamGroup) -> object:
    from controllers.exploration_faster import _TuningParams

    values = vector.tolist()
    updates: dict[str, object] = {}
    cursor = 0

    def take(count: int) -> list[float]:
        nonlocal cursor
        chunk = values[cursor : cursor + count]
        cursor += count
        return chunk

    if group in ("residual", "all"):
        throttle_row = tuple(take(15))
        steer_row = tuple(take(15))
        updates["actor_weights"] = (throttle_row, steer_row)
        scales = take(2)
        updates["action_residual_scales"] = (
            _clip(scales[0], RESIDUAL_SCALE_BOUNDS),
            _clip(scales[1], RESIDUAL_SCALE_BOUNDS),
        )
        updates["hazard_residual_scale_max"] = _clip(take(1)[0], HAZARD_SCALE_BOUNDS)
    if group in ("line", "all"):
        updates["launch_corner_offsets_m"] = tuple(_clip(v, CORNER_OFFSET_BOUNDS) for v in take(5))
        updates["launch_curve_speed_gains"] = tuple(_clip(v, CURVE_GAIN_BOUNDS) for v in take(5))
        updates["launch_straight_speed_mps"] = _clip(take(1)[0], STRAIGHT_SPEED_BOUNDS)
        updates["recurring_corner_offsets_m"] = tuple(_clip(v, CORNER_OFFSET_BOUNDS) for v in take(5))
        updates["recurring_curve_speed_gains"] = tuple(_clip(v, CURVE_GAIN_BOUNDS) for v in take(5))
        updates["recurring_straight_speed_mps"] = _clip(take(1)[0], STRAIGHT_SPEED_BOUNDS)

    return replace(base, **updates) if isinstance(base, _TuningParams) else _TuningParams(**updates)


def evaluate_candidate(
    tuning: object, *, seeds: tuple[int, ...], round_seconds: float, focus_sector: int | None
) -> tuple[float, float]:
    """Return (mean_focus_weighted_reward_per_tick, mean_damage) over `seeds`."""
    from controllers.dynamics.hybrid_telemetry import Episode, TelemetryController
    from controllers.exploration_faster import create_controller
    from racing import run_headless_head_to_head
    from racing.race.rules import HeadToHeadRaceRules

    def factory() -> object:
        return create_controller(tuning)

    episode_rewards: list[float] = []
    episode_damages: list[float] = []
    for seed in seeds:
        episodes: list[Episode] = []
        challenger = TelemetryController(factory, episodes=episodes)
        incumbent = TelemetryController(factory, episodes=episodes)
        run_headless_head_to_head(
            challenger_controller=challenger,
            incumbent_controller=incumbent,
            race_count=1,
            round_seconds=round_seconds,
            random_seed=seed,
            rules=HeadToHeadRaceRules(marshal_enabled=False),
        )
        for episode in episodes:
            if not episode.ticks:
                continue
            weighted_rewards = [
                tick.reward * (3.0 if focus_sector is not None and tick.sector == focus_sector else 1.0)
                for tick in episode.ticks
            ]
            episode_rewards.append(statistics.fmean(weighted_rewards))
            episode_damages.append(episode.ticks[-1].damage)

    mean_reward = statistics.fmean(episode_rewards) if episode_rewards else -1e9
    mean_damage = statistics.fmean(episode_damages) if episode_damages else 1.0
    return mean_reward, mean_damage


def train(
    *,
    generations: int,
    population: int,
    elite_fraction: float,
    seeds: tuple[int, ...],
    round_seconds: float,
    group: ParamGroup,
    focus_sector: int | None,
    resume_path: Path | None,
    rng: np.random.Generator,
) -> tuple[object, float]:
    from controllers.exploration_faster import _TuningParams

    base = _load_tuning(resume_path) if resume_path is not None else _TuningParams()
    mean = flatten(base, group)
    sigma = sigma_vector(group)
    elite_count = max(1, round(population * elite_fraction))

    baseline_reward, baseline_damage = evaluate_candidate(
        base, seeds=seeds, round_seconds=round_seconds, focus_sector=focus_sector
    )
    print(f"gen 0 (baseline): reward={baseline_reward:.4f} damage={baseline_damage:.4f}")
    best_tuning, best_reward = base, baseline_reward

    for generation in range(1, generations + 1):
        candidates = mean[np.newaxis, :] + rng.normal(0.0, 1.0, size=(population, mean.shape[0])) * sigma[np.newaxis, :]
        scored: list[tuple[float, float, np.ndarray, object]] = []
        for candidate_vector in candidates:
            tuning = unflatten(candidate_vector, base, group)
            reward, damage = evaluate_candidate(
                tuning, seeds=seeds, round_seconds=round_seconds, focus_sector=focus_sector
            )
            if damage > baseline_damage + MAX_DAMAGE_REGRESSION:
                reward -= 1e6  # hard-reject: never let a damage regression win on reward alone
            scored.append((reward, damage, candidate_vector, tuning))

        scored.sort(key=lambda item: item[0], reverse=True)
        elites = scored[:elite_count]
        mean = np.mean(np.stack([vector for _reward, _damage, vector, _tuning in elites]), axis=0)

        top_reward, top_damage, _vector, top_tuning = scored[0]
        print(
            f"gen {generation}: best reward={top_reward:.4f} damage={top_damage:.4f} "
            f"(elite mean reward={statistics.fmean(r for r, _d, _v, _t in elites):.4f})"
        )
        if top_reward > best_reward and top_damage <= baseline_damage + MAX_DAMAGE_REGRESSION:
            best_tuning, best_reward = top_tuning, top_reward

    return best_tuning, best_reward


def _load_tuning(path: Path) -> object:
    from controllers.exploration_faster import _coerce_tuning_overrides, _TuningParams

    raw = json.loads(path.read_text(encoding="utf-8"))
    return _TuningParams(**_coerce_tuning_overrides(raw))


def _tuning_to_json(tuning: object) -> dict[str, object]:
    return {
        "actor_weights": [list(tuning.actor_weights[0]), list(tuning.actor_weights[1])],
        "action_residual_scales": list(tuning.action_residual_scales),
        "hazard_residual_scale_max": tuning.hazard_residual_scale_max,
        "launch_corner_offsets_m": list(tuning.launch_corner_offsets_m),
        "launch_curve_speed_gains": list(tuning.launch_curve_speed_gains),
        "launch_straight_speed_mps": tuning.launch_straight_speed_mps,
        "recurring_corner_offsets_m": list(tuning.recurring_corner_offsets_m),
        "recurring_curve_speed_gains": list(tuning.recurring_curve_speed_gains),
        "recurring_straight_speed_mps": tuning.recurring_straight_speed_mps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the hybrid controller's tuning (Phases 2-5).")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite-fraction", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--round-seconds", type=float, default=18.0)
    parser.add_argument("--params", choices=("residual", "line", "all"), default="all")
    parser.add_argument("--focus-sector", type=int, default=None, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--resume", type=Path, default=None, help="start from a previously trained tuning file")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the search itself")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    best_tuning, best_reward = train(
        generations=args.generations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        seeds=tuple(args.seeds),
        round_seconds=args.round_seconds,
        group=args.params,
        focus_sector=args.focus_sector,
        resume_path=args.resume,
        rng=rng,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_tuning_to_json(best_tuning), indent=2), encoding="utf-8")
    print(f"\nbest reward={best_reward:.4f} -- wrote {args.output}")
    print("Validate with, e.g.:")
    print(
        f"  uv run python scripts/evaluate_controller.py "
        f"--tuning-path {args.output} --report artifacts/eval_trained.json"
    )


if __name__ == "__main__":
    main()
