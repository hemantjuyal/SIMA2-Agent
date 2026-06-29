"""Defines the schema and supported actions for VizDoom environments."""
from gsima.schema import CanonicalAction

# VizdoomBasic-v1 exposes a small, discrete action set that maps well to the
# canonical action set used by the planner.
SUPPORTED_ACTIONS = [
    CanonicalAction.MOVE_LEFT,
    CanonicalAction.MOVE_RIGHT,
    CanonicalAction.SHOOT,
]
