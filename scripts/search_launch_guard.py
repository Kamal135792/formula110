"""Search first-lap transition policies with damage as a hard constraint."""

from __future__ import annotations

from statistics import mean

from train_v0 import SoloRaceRunner

from controllers.v41 import Controller as SectorController
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS

SEEDS = (10, 31, 44, 64, 68, 78, 86, 93, 97, 110)
TICKS = (0, 10, 20, 30, 40, 50, 60, 75, 90, 120)


def main() -> int:
    runner = SoloRaceRunner()
    rows: list[tuple[float, float, float, int]] = []
    try:
        for ticks in TICKS:
            results = [
                runner.run(
                    SectorController(
                        corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                        curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
                        speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
                        straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
                        initial_stable_ticks=ticks,
                        hazard_stable_ticks=ticks,
                        extended_start_hazards=True,
                    ),
                    seed=seed,
                    duration_seconds=5.0,
                )
                for seed in SEEDS
            ]
            total_damage = sum(result.damage for result in results)
            maximum_damage = max(result.damage for result in results)
            progress = mean(result.raw_progress_m for result in results)
            rows.append((total_damage, -progress, maximum_damage, ticks))
            print(
                f"ticks={ticks:>3} damage={total_damage:.6f} max={maximum_damage:.6f} "
                f"progress={progress:.2f} min={min(result.raw_progress_m for result in results):.2f}"
            )
    finally:
        runner.close()
    print("ranked (zero damage first):")
    for total_damage, negative_progress, maximum_damage, ticks in sorted(rows):
        print(
            f"ticks={ticks:>3} damage={total_damage:.6f} max={maximum_damage:.6f} "
            f"progress={-negative_progress:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
