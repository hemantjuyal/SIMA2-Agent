"""
Defines the global, canonical action space for all agents.

All possible actions any agent can take in any environment should be
defined here. Each environment will then select a subset of these actions.
"""
from enum import Enum

class CanonicalAction(Enum):
    # Basic navigation
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    SHOOT = "SHOOT"

    # Future potential actions for more complex environments
    # JUMP = "JUMP"
    # USE_ITEM = "USE_ITEM"
    # PICK_UP = "PICK_UP"
    # DROP_ITEM = "DROP_ITEM"
