"""Defines the schema and supported actions for VizDoom Deadly Corridor environment."""
from enum import Enum

class VizdoomDeadlyCorridorAction(Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    SHOOT = "SHOOT"
    STOP = "STOP"

SUPPORTED_ACTIONS = list(VizdoomDeadlyCorridorAction)
