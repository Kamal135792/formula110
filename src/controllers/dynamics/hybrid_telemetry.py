"""Shared headless-race telemetry plumbing for evaluating/training the hybrid controller.

Wraps `controllers.exploration_faster.Controller` (or any factory producing a
compatible controller) so that a single `run_headless_head_to_head` race
yields, per tick: track progress/sector (via the controller's own public-
sensor localizer -- no privileged track state), the emitted command, and the
Phase 3 RL-correction reward from `hybrid_reward.step_reward`. Both
`scripts/evaluate_controller.py` (Phase 1/6 baselining) and
`scripts/train_hybrid_residual.py` (Phase 2-5 training) build on this.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Reused directly rather than duplicated: both are the hybrid controller's own
# public-sensor-only progress/sector logic, kept as one definition so a future
# change to sector boundaries or localization can't silently drift between
# the controller and its training/evaluation tooling.
from controllers.dynamics.hybrid_reward import step_reward
from controllers.exploration_faster import _corner_sector, _TrackLocalizer  # pyright: ignore[reportPrivateUsage]
from racing import RobotCommand, RobotController, RobotSensors


@dataclass(slots=True)
class TickRecord:
    tick: int
    dt_s: float
    progress_m: float
    sector: int | None
    speed_mps: float
    throttle: float
    steer: float
    wall_contact_s: float
    damage: float
    reward: float


@dataclass(slots=True)
class Episode:
    """One car's telemetry for one race."""

    ticks: list[TickRecord] = field(default_factory=list[TickRecord])


class TelemetryController:
    """Wraps a controller factory, recording tick-level telemetry as it drives.

    `copy_for_car()` (called once per car per race by `run_headless_head_to_head`)
    creates a fresh wrapped instance with its own `Episode` and appends that
    episode to the shared `episodes` list, so after the race every car's full
    per-tick history is available in creation order.
    """

    def __init__(self, factory: Callable[[], RobotController], episodes: list[Episode] | None = None) -> None:
        self._factory = factory
        self._inner = factory()
        self._localizer = _TrackLocalizer()
        self.episodes: list[Episode] = episodes if episodes is not None else []
        self.episode = Episode()
        self._registered = False
        self._previous_command = RobotCommand()
        self._previous_damage = 0.0
        self._previous_progress_m: float | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        # `run_headless_head_to_head` always drives a fresh `copy_for_car()`
        # instance rather than the one it was handed, so registering eagerly
        # in `__init__` would leave a stray empty episode for every never-
        # driven top-level/prototype instance. Register lazily, once, here.
        if not self._registered:
            self.episodes.append(self.episode)
            self._registered = True

        command = self._inner(sensors)
        progress_m = self._localizer.update(sensors)
        total_length_m = self._localizer.total_length_m

        if sensors.tick == 0 or self._previous_progress_m is None:
            progress_delta_m = 0.0
        else:
            raw_delta = progress_m - self._previous_progress_m
            if raw_delta < -total_length_m / 2.0:
                raw_delta += total_length_m
            elif raw_delta > total_length_m / 2.0:
                raw_delta -= total_length_m
            progress_delta_m = raw_delta

        reward = step_reward(
            sensors=sensors,
            command=command,
            previous_command=self._previous_command,
            progress_delta_m=progress_delta_m,
            previous_damage=self._previous_damage,
        )
        self.episode.ticks.append(
            TickRecord(
                tick=sensors.tick,
                dt_s=sensors.dt_s,
                progress_m=progress_m,
                sector=_corner_sector(progress_m),
                speed_mps=sensors.odometry.speed_mps,
                throttle=command.throttle,
                steer=command.steer,
                wall_contact_s=sensors.contact.wall,
                damage=sensors.contact.damage,
                reward=reward,
            )
        )
        self._previous_command = command
        self._previous_damage = sensors.contact.damage
        self._previous_progress_m = progress_m
        return command

    def copy_for_car(self) -> TelemetryController:
        return TelemetryController(self._factory, episodes=self.episodes)


def sector_visit_durations(episode: Episode) -> dict[int, list[float]]:
    """Duration (seconds) of each contiguous run of ticks spent in one sector."""
    durations: dict[int, list[float]] = {}
    current_sector: int | None = None
    current_duration_s = 0.0
    for tick in episode.ticks:
        if tick.sector != current_sector:
            if current_sector is not None and current_duration_s > 0.0:
                durations.setdefault(current_sector, []).append(current_duration_s)
            current_sector = tick.sector
            current_duration_s = 0.0
        if tick.sector is not None:
            current_duration_s += tick.dt_s
    if current_sector is not None and current_duration_s > 0.0:
        durations.setdefault(current_sector, []).append(current_duration_s)
    return durations
