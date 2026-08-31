"""Identify which fine V45 sector creates the seed-7777 regression."""

from __future__ import annotations

from train_v0 import SoloRaceRunner

from controllers.v41 import Controller
from controllers.v42 import LEARNED_CORNER_OFFSETS_M
from controllers.v44 import FINE_CURVE_SPEED_GAINS
from controllers.v45 import LIMIT_CURVE_SPEED_GAINS


def main() -> int:
    cases: list[tuple[str, tuple[float, ...]]] = [
        ("v44", FINE_CURVE_SPEED_GAINS),
        ("v45", LIMIT_CURVE_SPEED_GAINS),
    ]
    for sector in range(4):
        gains = list(LIMIT_CURVE_SPEED_GAINS)
        gains[sector] = FINE_CURVE_SPEED_GAINS[sector]
        cases.append((f"revert-sector-{sector}", tuple(gains)))

    runner = SoloRaceRunner()
    try:
        for name, gains in cases:
            result = runner.run(
                Controller(corner_offsets_m=LEARNED_CORNER_OFFSETS_M, curve_speed_gains=gains),
                seed=7777,
                duration_seconds=40.0,
            )
            lap = "--" if result.best_lap_seconds is None else f"{result.best_lap_seconds:.3f}"
            print(
                f"{name:16s} gains={gains} progress={result.raw_progress_m:7.1f} "
                f"lap={lap} wall={result.wall_contact_seconds:5.2f} damage={result.damage:.3f}"
            )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
