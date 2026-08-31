"""Search first-pass racing-line guards and reject every damaging candidate."""

from __future__ import annotations

from statistics import mean

from train_v0 import SoloRaceRunner

from controllers.v64 import Controller

SEEDS = (10, 31, 44, 64, 68, 78, 86, 93, 97, 110)
STARTS_M = (35.0, 38.0, 40.0)
CORRECTIONS = (-0.10, -0.125, -0.15, -0.175, -0.20)


def main() -> int:
    runner = SoloRaceRunner()
    rows: list[tuple[float, float, float, float, float]] = []
    try:
        for start_m in STARTS_M:
            for correction in CORRECTIONS:
                peak_start_m = min(start_m + 6.0, 44.0)
                results = [
                    runner.run(
                        Controller(
                            guard_start_m=start_m,
                            guard_peak_start_m=peak_start_m,
                            guard_peak_end_m=48.0,
                            guard_end_m=48.0,
                            avoidance_steer=correction,
                        ),
                        seed=seed,
                        duration_seconds=5.0,
                    )
                    for seed in SEEDS
                ]
                total_damage = sum(result.damage for result in results)
                maximum_damage = max(result.damage for result in results)
                progress = mean(result.raw_progress_m for result in results)
                rows.append((total_damage, -progress, maximum_damage, start_m, correction))
                print(
                    f"start={start_m:>4.0f} correction={correction:+.3f} damage={total_damage:.6f} "
                    f"max={maximum_damage:.6f} progress={progress:.2f} "
                    f"min={min(result.raw_progress_m for result in results):.2f}"
                )
    finally:
        runner.close()
    print("ranked (zero damage first):")
    for total_damage, negative_progress, maximum_damage, start_m, correction in sorted(rows):
        print(
            f"start={start_m:>4.0f} correction={correction:+.3f} damage={total_damage:.6f} "
            f"max={maximum_damage:.6f} progress={-negative_progress:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
