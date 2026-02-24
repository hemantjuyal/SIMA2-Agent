"""Adapter for Gym-MiniGrid environments."""
import logging
from gsima.environments.base import BaseAdapter
from gsima.environments.minigrid.schema import SUPPORTED_ACTIONS
from gsima.schema import CanonicalAction

class MiniGridAdapter(BaseAdapter):
    """Action adapter for Gym-MiniGrid environments."""
    def __init__(self, env):
        super().__init__(env)
        # Dynamically build the action map from the supported actions schema
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        logging.debug(f"MiniGridAdapter initialized with action map: {self.action_map}")

    def _get_env_action(self, action: CanonicalAction):
        """Maps a canonical action to a specific MiniGrid environment action."""
        logging.debug(f"Mapping canonical action '{action}' to environment action.")
        if action == CanonicalAction.MOVE_FORWARD:
            return self.env.unwrapped.actions.forward
        if action == CanonicalAction.TURN_LEFT:
            return self.env.unwrapped.actions.left
        if action == CanonicalAction.TURN_RIGHT:
            return self.env.unwrapped.actions.right
        if action == CanonicalAction.STOP:
            return -1 # Special sentinel value
        logging.error(f"Unsupported canonical action for MiniGrid: {action}")
        raise ValueError(f"Unsupported action for MiniGrid: {action}")

    def translate_action(self, action_name: str) -> any:
        """Translates a canonical action name for MiniGrid."""
        logging.debug(f"Translating action name '{action_name}' for MiniGrid.")
        action = self.action_map.get(action_name)
        if action is None:
            logging.warning(f"Unknown action '{action_name}' for MiniGrid. Defaulting to STOP.")
            return self.action_map[CanonicalAction.STOP.name]
        logging.debug(f"Translated '{action_name}' to environment action '{action}'.")
        return action

    def get_canonical_actions(self) -> list:
        """Returns a list of canonical actions supported by the MiniGrid environment."""
        return SUPPORTED_ACTIONS
