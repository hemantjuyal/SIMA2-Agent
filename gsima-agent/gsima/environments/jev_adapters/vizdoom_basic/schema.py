"""Defines the schema and supported actions for VizDoom environments."""
from enum import Enum

class VizdoomAction(Enum):
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    SHOOT = "SHOOT"
    STOP = "STOP"

SUPPORTED_ACTIONS = list(VizdoomAction)
