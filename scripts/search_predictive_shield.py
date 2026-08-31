"""Grid-search V65's predictive safety shield on damage-sensitive starts."""

from __future__ import annotations

from statistics import mean

from train_v0 import SoloRaceRunner

from controllers.v65 import Controller

SEEDS = (10, 31, 44, 64, 68, 78, 86, 93, 97, 110)


def main() -> int:
    runner = SoloRaceRunner()
    rows: list[tuple[float, float, float, float, float]] = []
    try:
        for trigger in (2.5, 3.0, 3.5, 4.0):
            for steer in (-0.65, -0.8, -0.95):
                results = [
                    runner.run(
                        Controller(clearance_trigger_m=trigger, full_response_steer=steer),
                        seed=seed,
                        duration_seconds=5.0,
                    )
                    for seed in SEEDS
                ]
                damage = sum(result.damage for result in results)
                progress = mean(result.raw_progress_m for result in results)
                maximum = max(result.damage for result in results)
                rows.append((damage, -progress, maximum, trigger, steer))
                print(
                    f"trigger={trigger:.1f} steer={steer:+.2f} damage={damage:.6f} max={maximum:.6f} "
                    f"progress={progress:.2f} min={min(result.raw_progress_m for result in results):.2f}"
                )
    finally:
        runner.close()
    print("ranked (zero damage first):")
    for damage, negative_progress, maximum, trigger, steer in sorted(rows):
        print(
            f"trigger={trigger:.1f} steer={steer:+.2f} damage={damage:.6f} max={maximum:.6f} "
            f"progress={-negative_progress:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
