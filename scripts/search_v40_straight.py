"""Search the localized straight attack without changing the base controller."""

from __future__ import annotations

from benchmark_controller import summarize
from train_v0 import SoloRaceRunner

from controllers.v40 import Controller

SEEDS = (110, 2026, 42, 73, 901, 1337, 13, 21)
BOOST_ENDS_M = (0.0, 2.0, 4.0, 6.0, 8.0)
TARGET_SPEEDS_MPS = (35.5, 36.5, 37.5)


def main() -> int:
    runner = SoloRaceRunner()
    try:
        for target_speed in TARGET_SPEEDS_MPS:
            for boost_end in BOOST_ENDS_M:
                results = [
                    runner.run(
                        Controller(boost_end_m=boost_end, boost_target_speed_mps=target_speed),
                        seed=seed,
                        duration_seconds=24.0,
                    )
                    for seed in SEEDS
                ]
                summary = summarize(results)
                lap = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.3f}"
                print(
                    f"target={target_speed:4.1f} end={boost_end:4.1f} "
                    f"score={summary.score:7.2f} min={summary.minimum_progress_m:6.1f} "
                    f"lap={lap} speed={summary.mean_max_speed_mps:5.2f} "
                    f"wall={summary.total_wall_contact_seconds:5.2f} off={summary.total_off_track_seconds:6.2f}",
                    flush=True,
                )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
