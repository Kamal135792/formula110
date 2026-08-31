"""Evolutionary joint search over V51's partition policy parameters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

from benchmark_controller import summarize
from train_v0 import EvaluationSummary, SoloEpisodeResult, SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v51 import EARLY_BRAKING_BOUNDARIES_M, HIGH_STRAIGHT_SPEED_MPS

SEEDS = (110, 2026, 42, 73, 500, 901, 7777)
GENERATIONS = 4
CANDIDATES_PER_GENERATION = 6
OUTPUT_PATH = Path("artifacts/evolved_v52_partitions.json")


@dataclass(frozen=True, slots=True)
class Genome:
    corner_offsets_m: tuple[float, ...]
    curve_speed_gains: tuple[float, ...]
    speed_boundaries_m: tuple[float, ...]
    straight_speed_mps: float


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _mutate(parent: Genome, random: Random, scale: float) -> Genome:
    offsets = tuple(_clamp(value + random.gauss(0.0, 0.14 * scale), 0.25, 2.25) for value in parent.corner_offsets_m)
    gains = tuple(
        _clamp(value + random.gauss(0.0, (0.07 if index == 0 else 0.055) * scale), 0.32, 1.40)
        for index, value in enumerate(parent.curve_speed_gains)
    )
    boundaries = list(parent.speed_boundaries_m)
    boundaries[0] = _clamp(boundaries[0] + random.gauss(0.0, 1.5 * scale), 1.0, 11.0)
    for index in range(1, 6):
        boundaries[index] += random.gauss(0.0, 1.2 * scale)
    straight_speed = _clamp(parent.straight_speed_mps + random.gauss(0.0, 0.35 * scale), 36.0, 39.0)
    return Genome(offsets, gains, tuple(boundaries), straight_speed)


def _evaluate(runner: SoloRaceRunner, genome: Genome) -> EvaluationSummary:
    results: list[SoloEpisodeResult] = []
    for seed in SEEDS:
        controller = Controller(
            corner_offsets_m=genome.corner_offsets_m,
            curve_speed_gains=genome.curve_speed_gains,
            speed_boundaries_m=genome.speed_boundaries_m,
            straight_speed_numerator=genome.straight_speed_mps,
        )
        results.append(runner.run(controller, seed=seed, duration_seconds=28.0))
    return summarize(results)


def _fitness(summary: EvaluationSummary) -> float:
    safety_penalty = 10.0 * max(0.0, 600.0 - summary.minimum_progress_m)
    return summary.mean_progress_m + 0.08 * summary.score - safety_penalty


def main() -> int:
    random = Random(51_110)
    incumbent = Genome(
        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
        curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
        speed_boundaries_m=EARLY_BRAKING_BOUNDARIES_M,
        straight_speed_mps=HIGH_STRAIGHT_SPEED_MPS,
    )
    runner = SoloRaceRunner()
    try:
        incumbent_summary = _evaluate(runner, incumbent)
        incumbent_fitness = _fitness(incumbent_summary)
        print(f"baseline fitness={incumbent_fitness:.2f} progress={incumbent_summary.mean_progress_m:.1f}")
        for generation in range(1, GENERATIONS + 1):
            scale = 0.72 ** (generation - 1)
            generation_best = (incumbent_fitness, incumbent, incumbent_summary)
            for candidate_index in range(CANDIDATES_PER_GENERATION):
                candidate = _mutate(incumbent, random, scale)
                summary = _evaluate(runner, candidate)
                fitness = _fitness(summary)
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"generation={generation} candidate={candidate_index} fitness={fitness:7.2f} "
                    f"progress={summary.mean_progress_m:6.1f} score={summary.score:7.2f} "
                    f"min={summary.minimum_progress_m:6.1f} lap={lap}",
                    flush=True,
                )
                if fitness > generation_best[0]:
                    generation_best = (fitness, candidate, summary)
            incumbent_fitness, incumbent, incumbent_summary = generation_best
            print(f"accepted generation={generation}: fitness={incumbent_fitness:.2f} genome={incumbent}", flush=True)
    finally:
        runner.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "fitness": incumbent_fitness,
                "genome": asdict(incumbent),
                "summary": asdict(incumbent_summary),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
