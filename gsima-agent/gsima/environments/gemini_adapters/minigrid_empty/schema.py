"""Defines the schema and supported actions for MiniGrid environments."""
from enum import Enum

class MiniGridAction(Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"

# This list defines which actions are supported in MiniGrid.
SUPPORTED_ACTIONS = list(MiniGridAction)
