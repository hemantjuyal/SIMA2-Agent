"""Defines the schema and supported actions for BabyAI UnlockPickup environment."""
from enum import Enum

class BabyAIAction(Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    PICK_UP = "PICK_UP"
    DROP = "DROP"
    TOGGLE = "TOGGLE"
    STOP = "STOP"

# This list defines which actions are supported in BabyAI.
SUPPORTED_ACTIONS = list(BabyAIAction)
