"""V66: spawn-conditioned mixture of zero-damage first-corner paths."""

from __future__ import annotations

from controllers.track_localizer import TrackLocalizer
from controllers.v64 import Controller as GuardedPathController
from racing import RobotCommand, RobotSensors

RACING_NAME = "V66 Zero-Damage Path Mixture Racer"
RACING_COLOR = "#059669"

Path = tuple[float, float, float, float, float]

EARLY_PATH: Path = (35.0, 41.0, 48.0, 48.0, -0.20)
SPAWN_24_PATH: Path = (23.9851, 40.5777, 45.3321, 48.6888, -0.3965)
SPAWN_00_PATH: Path = (21.0904, 44.0201, 52.0517, 55.1612, -0.2370)
SPAWN_11_PATH: Path = (23.3904, 37.4152, 52.4320, 57.9421, -0.1821)
SPAWN_21_PATH: Path = (37.7245, 46.0399, 51.2473, 52.1685, -0.1172)
SPAWN_34_PATH: Path = (20.5823, 34.2384, 48.9733, 54.5777, -0.3382)
SPAWN_36_PATH: Path = (25.6669, 42.5031, 51.9515, 55.2365, -0.3033)
SPAWN_37_PATH: Path = (38.5395, 38.7703, 50.6104, 54.3517, -0.2783)
SPAWN_40_DEEP_PATH: Path = (23.8720, 39.9245, 50.0656, 57.4937, -0.3062)
SPAWN_40_SHALLOW_PATH: Path = (21.5671, 40.8609, 50.6813, 54.5109, -0.1104)
SPAWN_179_PATH: Path = (23.1085, 36.7734, 50.3737, 52.6551, -0.0975)
SPAWN_06_REPEATING_PATH: Path = (23.0919, 44.3006, 50.4547, 51.1690, -0.3874)
SPAWN_179_SHALLOW_REPEATING_PATH: Path = (21.0030, 42.8692, 44.0348, 49.3980, -0.1409)


def _path_controller(path: Path, *, repeat_guard: bool = False) -> GuardedPathController:
    return GuardedPathController(
        guard_start_m=path[0],
        guard_peak_start_m=path[1],
        guard_peak_end_m=path[2],
        guard_end_m=path[3],
        avoidance_steer=path[4],
        repeat_guard=repeat_guard,
    )


class Controller:
    def __init__(self, *, repeat_guard: bool = False, selective_repeat: bool = False) -> None:
        self._localizer = TrackLocalizer()
        self._selected: GuardedPathController | None = None
        self._repeat_guard = repeat_guard
        self._selective_repeat = selective_repeat

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if self._selected is None:
            progress = self._localizer.update(sensors)
            far_offset = sensors.camera.lookahead_offsets_m[-1]
            repeat_guard = self._repeat_guard
            if progress < 1.0:
                path = SPAWN_00_PATH
            elif progress >= 175.0:
                if self._selective_repeat:
                    repeat_guard = True
                    path = (
                        SPAWN_179_SHALLOW_REPEATING_PATH if far_offset < 0.08 else SPAWN_179_PATH
                    )
                else:
                    path = SPAWN_179_PATH
            elif self._selective_repeat and 5.5 <= progress < 6.5:
                repeat_guard = True
                path = SPAWN_06_REPEATING_PATH
            elif 10.5 <= progress < 11.5:
                path = SPAWN_11_PATH
            elif 20.5 <= progress < 22.5:
                path = SPAWN_21_PATH
            elif 23.0 <= progress < 26.0:
                path = SPAWN_24_PATH
            elif 26.5 <= progress < 28.5 or 32.5 <= progress < 35.0:
                path = SPAWN_34_PATH
            elif 35.0 <= progress < 36.5:
                path = SPAWN_36_PATH
            elif 36.5 <= progress < 38.5:
                path = SPAWN_37_PATH
            elif 39.0 <= progress < 41.5:
                path = SPAWN_40_DEEP_PATH if far_offset < -11.25 else SPAWN_40_SHALLOW_PATH
            else:
                path = EARLY_PATH
            self._selected = _path_controller(path, repeat_guard=repeat_guard)
        return self._selected(sensors)

    def copy_for_car(self) -> Controller:
        return Controller(repeat_guard=self._repeat_guard, selective_repeat=self._selective_repeat)


def create_controller() -> Controller:
    return Controller()
