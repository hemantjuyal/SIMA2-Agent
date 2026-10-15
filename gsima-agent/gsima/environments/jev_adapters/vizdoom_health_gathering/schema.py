"""Defines the schema and supported actions for VizDoom Health Gathering environment."""
from enum import Enum

class VizdoomHealthGatheringAction(Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"

SUPPORTED_ACTIONS = list(VizdoomHealthGatheringAction)
