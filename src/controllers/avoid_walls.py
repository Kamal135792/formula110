"""Self-driving robotic race car controller demo."""

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Avoid Walls"
RACING_COLOR: str = "#ff00d9"

def control(sensors: RobotSensors) -> RobotCommand:
    throttle: float = 0.4
    steer: float = 0.0
    return RobotCommand(throttle, steer)
