"""Tune stabilization separately for the two hazardous spawn regions."""

from __future__ import annotations

from train_v0 import SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v46 import ROBUST_LIMIT_CURVE_SPEED_GAINS
from controllers.v50 import AGGRESSIVE_BRAKING_BOUNDARIES_M, AGGRESSIVE_STRAIGHT_SPEED_MPS

HAZARD_SEEDS = (10, 13, 21)
STABILIZATION_TICKS = (60, 90, 120, 150, 180, 240, 300)


def main() -> int:
    runner = SoloRaceRunner()
    try:
        for ticks in STABILIZATION_TICKS:
            fields: list[str] = []
            for seed in HAZARD_SEEDS:
                controller = Controller(
                    corner_offsets_m=LEARNED_CORNER_OFFSETS_M,
                    curve_speed_gains=ROBUST_LIMIT_CURVE_SPEED_GAINS,
                    speed_boundaries_m=AGGRESSIVE_BRAKING_BOUNDARIES_M,
                    straight_speed_numerator=AGGRESSIVE_STRAIGHT_SPEED_MPS,
                    hazard_stable_ticks=ticks,
                )
                result = runner.run(controller, seed=seed, duration_seconds=40.0)
                lap = "--" if result.best_lap_seconds is None else f"{result.best_lap_seconds:.3f}"
                fields.append(
                    f"seed={seed}:progress={result.raw_progress_m:.1f},lap={lap},wall={result.wall_contact_seconds:.2f}"
                )
            print(f"ticks={ticks:3d} " + " | ".join(fields), flush=True)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
