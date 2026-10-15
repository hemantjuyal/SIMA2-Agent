"""Defines the schema and supported actions for VizDoom environments."""
from enum import Enum

class VizdoomPredictAction(Enum):
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    SHOOT = "SHOOT"
    STOP = "STOP"

# This list defines which actions are supported in ViZDoom.
SUPPORTED_ACTIONS = list(VizdoomPredictAction)
