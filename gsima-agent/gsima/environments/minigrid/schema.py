"""Defines the schema and supported actions for MiniGrid environments."""
from gsima.schema import CanonicalAction

# This list defines which canonical actions are supported in MiniGrid.
SUPPORTED_ACTIONS = [
    CanonicalAction.MOVE_FORWARD,
    CanonicalAction.TURN_LEFT,
    CanonicalAction.TURN_RIGHT,
    CanonicalAction.STOP,
]
