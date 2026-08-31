"""V37: track-localized stable launch and apex partitions without boost."""

from __future__ import annotations

from controllers.v36 import Controller as PartitionController

RACING_NAME = "V37 Partition Apex Racer"
RACING_COLOR = "#18ffff"


class Controller(PartitionController):
    def __init__(self) -> None:
        super().__init__(boost_enabled=False)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
