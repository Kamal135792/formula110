"""Train and evaluate ``controllers.v0`` with repeated headless solo races.

The trainer may use simulator progress as a privileged reward signal.  The
saved controller still observes only the public ``RobotSensors`` contract.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, cast

from controllers.v0 import (
    Controller,
    LearningMetrics,
    PlannerConfig,
    PolicyParameters,
    TrainingStep,
    default_policy_parameters,
    improve_policy,
    load_policy_parameters,
    save_policy_parameters,
)
from racing import RobotController
from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
)
from racing.race.progress import default_track_progress_model, project_track_position
from racing.race.runtime import (
    RaceCarRuntime,
    lap_progress_tracker_for_spawn_pose,
    race_contact_states,
    race_spawn_poses,
    robot_is_eliminated,
    robot_score_damage,
    robot_track_point,
    update_race_runtime_after_step,
)
from racing.race.sensors import RobotSensorBuilderState, build_robot_sensors

FIXED_DELTA_SECONDS = 1.0 / 60.0
DEFAULT_CHECKPOINT_PATH = Path(__file__).parents[1] / "src" / "controllers" / "v0_weights.json"
DEFAULT_LOG_PATH = Path(__file__).parents[1] / "artifacts" / "v0_training_log.json"


@dataclass(frozen=True, slots=True)
class SoloEpisodeResult:
    seed: int
    elapsed_seconds: float
    raw_progress_m: float
    lap_count: int
    best_lap_seconds: float | None
    max_speed_mps: float
    damage: float
    wall_contact_seconds: float
    off_track_seconds: float
    eliminated: bool

    @property
    def evaluation_score(self) -> float:
        """Favor progress while making damage and failures expensive."""
        return (
            self.raw_progress_m
            - 140.0 * self.damage
            - 6.0 * self.wall_contact_seconds
            - 50.0 * self.off_track_seconds
            - (100.0 if self.eliminated else 0.0)
        )

    @property
    def terminal_training_reward(self) -> float:
        """Add the simulator's official outcome to sensor-derived rewards."""
        return (
            0.20 * self.raw_progress_m
            - 100.0 * self.damage
            - 2.0 * self.wall_contact_seconds
            - 20.0 * self.off_track_seconds
            - (60.0 if self.eliminated else 0.0)
        )


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    score: float
    mean_progress_m: float
    minimum_progress_m: float
    total_laps: int
    best_lap_seconds: float | None
    mean_max_speed_mps: float
    mean_damage: float
    total_wall_contact_seconds: float
    total_off_track_seconds: float
    elimination_count: int


class SoloRaceRunner:
    """Reusable Panda3D process hosting isolated single-car race episodes."""

    def __init__(self) -> None:
        configure_headless_panda()
        showbase = cast(Any, import_module("direct.showbase.ShowBase"))
        self._base = showbase.ShowBase(windowType="none")
        self._model = default_track_progress_model()

    def close(self) -> None:
        self._base.destroy()

    def run(self, controller: RobotController, *, seed: int, duration_seconds: float) -> SoloEpisodeResult:
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = self._base.render.attachNewNode(f"v0-training-{seed}")
        add_racing_scene_collisions(physics_world=physics_world, render=root)
        spawn_pose = race_spawn_poses(
            1,
            model=self._model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=seed,
            race_index=1,
        )[0]
        robot = create_robot_vehicle(
            world=physics_world,
            render=root,
            name=f"v0-training-car-{seed}",
            position=spawn_pose.position,
            heading_degrees=spawn_pose.heading_degrees,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        )
        physics_scene.vehicles.append(robot)
        runtime = RaceCarRuntime(
            robot=robot,
            tracker=lap_progress_tracker_for_spawn_pose(model=self._model, spawn_pose=spawn_pose),
        )
        elapsed_seconds = 0.0
        sensor_state = RobotSensorBuilderState()
        try:
            while elapsed_seconds < duration_seconds and not robot_is_eliminated(robot):
                sensors, sensor_state = build_robot_sensors(
                    physics_world=physics_world,
                    robot=robot,
                    track_model=self._model,
                    time_s=elapsed_seconds,
                    dt_s=FIXED_DELTA_SECONDS,
                    previous_state=sensor_state,
                )
                apply_robot_vehicle_command(robot=robot, command=controller(sensors))
                physics_scene.step(FIXED_DELTA_SECONDS)
                elapsed_seconds += FIXED_DELTA_SECONDS
                contact_state = race_contact_states(physics_world=physics_world, runtimes=(runtime,))[0]
                apply_wall_impact_damage(
                    physics_world=physics_world,
                    robots=(robot,),
                    fixed_time_step=physics_scene.fixed_time_step,
                )
                projection = project_track_position(self._model, robot_track_point(robot))
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=elapsed_seconds,
                    delta_seconds=FIXED_DELTA_SECONDS,
                )

            return SoloEpisodeResult(
                seed=seed,
                elapsed_seconds=elapsed_seconds,
                raw_progress_m=runtime.tracker.best_distance_m,
                lap_count=runtime.tracker.lap_count,
                best_lap_seconds=min(runtime.tracker.lap_times_seconds) if runtime.tracker.lap_times_seconds else None,
                max_speed_mps=runtime.max_speed_mps,
                damage=robot_score_damage(robot),
                wall_contact_seconds=runtime.tracker.wall_contact_seconds,
                off_track_seconds=runtime.off_track_seconds,
                eliminated=robot_is_eliminated(robot),
            )
        finally:
            root.removeNode()


def evaluate_policy(
    runner: SoloRaceRunner,
    parameters: PolicyParameters,
    *,
    seeds: tuple[int, ...],
    duration_seconds: float,
    planner_config: PlannerConfig | None = None,
) -> tuple[EvaluationSummary, list[SoloEpisodeResult]]:
    results = [
        runner.run(
            Controller(parameters.copy(), planner_config=planner_config),
            seed=seed,
            duration_seconds=duration_seconds,
        )
        for seed in seeds
    ]
    lap_times = [result.best_lap_seconds for result in results if result.best_lap_seconds is not None]
    summary = EvaluationSummary(
        score=mean(result.evaluation_score for result in results),
        mean_progress_m=mean(result.raw_progress_m for result in results),
        minimum_progress_m=min(result.raw_progress_m for result in results),
        total_laps=sum(result.lap_count for result in results),
        best_lap_seconds=min(lap_times) if lap_times else None,
        mean_max_speed_mps=mean(result.max_speed_mps for result in results),
        mean_damage=mean(result.damage for result in results),
        total_wall_contact_seconds=sum(result.wall_contact_seconds for result in results),
        total_off_track_seconds=sum(result.off_track_seconds for result in results),
        elimination_count=sum(result.eliminated for result in results),
    )
    return summary, results


def _training_batch(
    runner: SoloRaceRunner,
    parameters: PolicyParameters,
    *,
    random: Random,
    episode_count: int,
    duration_seconds: float,
    planner_config: PlannerConfig | None = None,
) -> tuple[list[list[TrainingStep]], list[SoloEpisodeResult]]:
    episodes: list[list[TrainingStep]] = []
    results: list[SoloEpisodeResult] = []
    for _ in range(episode_count):
        environment_seed = random.randrange(1, 10_000_000)
        exploration_seed = random.randrange(1, 10_000_000)
        controller = Controller(
            parameters.copy(),
            training=True,
            random_seed=exploration_seed,
            planner_config=planner_config,
        )
        result = runner.run(controller, seed=environment_seed, duration_seconds=duration_seconds)
        episodes.append(controller.finish_episode(result.terminal_training_reward))
        results.append(result)
    return episodes, results


def _parse_seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one evaluation seed is required")
    return seeds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--episodes-per-iteration", type=int, default=4)
    parser.add_argument(
        "--candidates-per-iteration",
        type=int,
        default=3,
        help="independent policy updates to train and evaluate before selecting the best",
    )
    parser.add_argument("--training-seconds", type=float, default=18.0)
    parser.add_argument("--evaluation-seconds", type=float, default=30.0)
    parser.add_argument("--eval-seeds", type=_parse_seeds, default=(110, 2026))
    parser.add_argument("--actor-learning-rate", type=float, default=0.018)
    parser.add_argument("--critic-learning-rate", type=float, default=0.035)
    parser.add_argument("--random-seed", type=int, default=110)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="initialize from this policy when --checkpoint does not exist",
    )
    parser.add_argument("--planner-config", type=Path, help="JSON file containing a planner object")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.iterations < 0:
        raise ValueError("iterations cannot be negative")
    if args.episodes_per_iteration < 1:
        raise ValueError("episodes-per-iteration must be positive")
    if args.candidates_per_iteration < 1:
        raise ValueError("candidates-per-iteration must be positive")
    if args.training_seconds <= 0.0 or args.evaluation_seconds <= 0.0:
        raise ValueError("episode durations must be positive")


def _print_evaluation(label: str, summary: EvaluationSummary) -> None:
    lap_text = "--" if summary.best_lap_seconds is None else f"{summary.best_lap_seconds:.2f}s"
    print(
        f"{label}: score={summary.score:.2f}, progress={summary.mean_progress_m:.1f}m "
        f"(min {summary.minimum_progress_m:.1f}m), laps={summary.total_laps}, best_lap={lap_text}, "
        f"speed={summary.mean_max_speed_mps:.2f}m/s, damage={summary.mean_damage:.3f}, "
        f"wall={summary.total_wall_contact_seconds:.2f}s, eliminated={summary.elimination_count}"
    )


def _iteration_record(
    *,
    iteration: int,
    accepted: bool,
    learning: LearningMetrics,
    training_results: list[SoloEpisodeResult],
    evaluation: EvaluationSummary,
    evaluation_results: list[SoloEpisodeResult],
    selected_candidate: int,
    effective_actor_learning_rate: float,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "accepted": accepted,
        "selected_candidate": selected_candidate,
        "effective_actor_learning_rate": effective_actor_learning_rate,
        "candidates": candidates,
        "learning": asdict(learning),
        "training": {
            "mean_progress_m": mean(result.raw_progress_m for result in training_results),
            "mean_damage": mean(result.damage for result in training_results),
            "elimination_count": sum(result.eliminated for result in training_results),
            "episodes": [asdict(result) for result in training_results],
        },
        "evaluation": asdict(evaluation),
        "evaluation_episodes": [asdict(result) for result in evaluation_results],
    }


def main() -> int:
    args = _parser().parse_args()
    _validate_arguments(args)
    checkpoint_path = cast(Path, args.checkpoint)
    log_path = cast(Path, args.log)
    eval_seeds = cast(tuple[int, ...], args.eval_seeds)
    if args.fresh:
        parameters = default_policy_parameters()
    elif checkpoint_path.exists():
        parameters = load_policy_parameters(checkpoint_path)
    elif args.initial_checkpoint is not None:
        parameters = load_policy_parameters(cast(Path, args.initial_checkpoint))
    else:
        parameters = default_policy_parameters()
    planner_config: PlannerConfig | None = None
    if args.planner_config is not None:
        planner_record = cast(
            dict[str, object],
            json.loads(cast(Path, args.planner_config).read_text(encoding="utf-8")),
        )
        planner_config = PlannerConfig(**cast(dict[str, Any], planner_record["planner"]))
    random = Random(args.random_seed)
    log_records: list[dict[str, object]] = []

    runner = SoloRaceRunner()
    try:
        baseline, baseline_results = evaluate_policy(
            runner,
            parameters,
            seeds=eval_seeds,
            duration_seconds=args.evaluation_seconds,
            planner_config=planner_config,
        )
        _print_evaluation("iteration 0", baseline)
        best_score = baseline.score
        parameters.best_evaluation_score = best_score
        log_records.append(
            {
                "iteration": 0,
                "accepted": True,
                "evaluation": asdict(baseline),
                "evaluation_episodes": [asdict(result) for result in baseline_results],
            }
        )
        if args.evaluate_only:
            return 0

        rejection_streak = 0
        for iteration in range(1, args.iterations + 1):
            learning_rate_scale = max(0.15, 0.65 ** (rejection_streak // 3))
            effective_actor_learning_rate = args.actor_learning_rate * learning_rate_scale
            attempts: list[
                tuple[
                    PolicyParameters,
                    LearningMetrics,
                    list[SoloEpisodeResult],
                    EvaluationSummary,
                    list[SoloEpisodeResult],
                ]
            ] = []
            candidate_records: list[dict[str, object]] = []
            for candidate_index in range(args.candidates_per_iteration):
                episodes, training_results = _training_batch(
                    runner,
                    parameters,
                    random=random,
                    episode_count=args.episodes_per_iteration,
                    duration_seconds=args.training_seconds,
                    planner_config=planner_config,
                )
                candidate, learning = improve_policy(
                    parameters,
                    episodes,
                    actor_learning_rate=effective_actor_learning_rate,
                    critic_learning_rate=args.critic_learning_rate,
                )
                evaluation, evaluation_results = evaluate_policy(
                    runner,
                    candidate,
                    seeds=eval_seeds,
                    duration_seconds=args.evaluation_seconds,
                    planner_config=planner_config,
                )
                attempts.append((candidate, learning, training_results, evaluation, evaluation_results))
                candidate_records.append(
                    {
                        "candidate": candidate_index + 1,
                        "evaluation": asdict(evaluation),
                        "learning": asdict(learning),
                    }
                )
                _print_evaluation(
                    f"iteration {iteration} candidate {candidate_index + 1}",
                    evaluation,
                )

            selected_index = max(range(len(attempts)), key=lambda index: attempts[index][3].score)
            candidate, learning, training_results, evaluation, evaluation_results = attempts[selected_index]
            accepted = evaluation.score > best_score
            if accepted:
                parameters = candidate
                best_score = evaluation.score
                rejection_streak = 0
            else:
                # Retain the improving critic and iteration count, but roll back
                # the actor so a noisy batch cannot damage the saved racer.
                parameters.critic_weights = candidate.critic_weights
                parameters.training_iterations = candidate.training_iterations
                rejection_streak += 1
            parameters.best_evaluation_score = best_score
            save_policy_parameters(parameters, checkpoint_path)
            outcome = "accepted" if accepted else "rejected"
            print(
                f"iteration {iteration}: candidate {selected_index + 1} selected and {outcome}; "
                f"best score={best_score:.2f}, actor_lr={effective_actor_learning_rate:.6f}"
            )
            log_records.append(
                _iteration_record(
                    iteration=iteration,
                    accepted=accepted,
                    learning=learning,
                    training_results=training_results,
                    evaluation=evaluation,
                    evaluation_results=evaluation_results,
                    selected_candidate=selected_index + 1,
                    effective_actor_learning_rate=effective_actor_learning_rate,
                    candidates=candidate_records,
                )
            )
    finally:
        runner.close()
        if not args.evaluate_only:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(log_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"saved best policy to {checkpoint_path}")
    print(f"saved training log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
