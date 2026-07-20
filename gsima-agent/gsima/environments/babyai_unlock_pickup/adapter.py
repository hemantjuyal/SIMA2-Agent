"""Adapter for BabyAI UnlockPickup environment."""
import logging
import copy
from typing import Dict, Any
import numpy as np

from gsima.environments.base import BaseAdapter
from gsima.environments.babyai_unlock_pickup.schema import SUPPORTED_ACTIONS, BabyAIAction

class BabyaiUnlockPickupAdapter(BaseAdapter):
    """Action adapter for BabyAI UnlockPickup environment."""
    def __init__(self, env):
        super().__init__(env)
        # Dynamically build the action map from the supported actions schema
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        logging.debug(f"BabyaiUnlockPickupAdapter initialized with action map: {self.action_map}")
        self.last_known_state = None

    def _get_env_action(self, action: BabyAIAction):
        """Maps a local action to a specific MiniGrid environment action."""
        logging.debug(f"Mapping action '{action}' to environment action.")
        if action == BabyAIAction.MOVE_FORWARD:
            return self.env.unwrapped.actions.forward
        if action == BabyAIAction.TURN_LEFT:
            return self.env.unwrapped.actions.left
        if action == BabyAIAction.TURN_RIGHT:
            return self.env.unwrapped.actions.right
        if action == BabyAIAction.PICK_UP:
            return self.env.unwrapped.actions.pickup
        if action == BabyAIAction.DROP:
            return self.env.unwrapped.actions.drop
        if action == BabyAIAction.TOGGLE:
            return self.env.unwrapped.actions.toggle
        if action == BabyAIAction.STOP:
            return -1 # Special sentinel value
        logging.error(f"Unsupported action for BabyAI: {action}")
        raise ValueError(f"Unsupported action for BabyAI: {action}")

    def translate_action(self, action_name: str) -> any:
        """Translates an action name for BabyAI."""
        logging.debug(f"Translating action name '{action_name}' for BabyAI.")
        action = self.action_map.get(action_name)
        if action is None:
            logging.warning(f"Unknown action '{action_name}' for BabyAI. Defaulting to STOP.")
            return self.action_map[BabyAIAction.STOP.name]
        logging.debug(f"Translated '{action_name}' to environment action '{action}'.")
        return action

    def get_canonical_actions(self) -> list:
        return SUPPORTED_ACTIONS

    def get_env_metadata(self) -> Dict[str, Any]:
        return {
            "env_type": "babyai_unlock_pickup",
            "supports_distance": True,
            "supports_stop": True,
            "state_keys": ["agent_pos", "agent_dir", "carrying"],
        }

    def get_action_metadata(self) -> Dict[str, Any]:
        return {
            action.name: {
                "canonical": action.name,
                "supports_stop": action.name == "STOP",
            }
            for action in SUPPORTED_ACTIONS
        }

    def get_state_summary(self) -> Dict[str, Any]:
        current_state = self.get_current_env_state()
        return {
            "agent_pos": current_state.get("agent_pos"),
            "agent_dir": current_state.get("agent_dir"),
            "carrying": current_state.get("carrying"),
        }

    def get_current_env_state(self) -> Dict[str, Any]:
        true_env = self.env.unwrapped
        carrying_type = true_env.carrying.type if true_env.carrying else None
        return {
            "agent_pos": np.array(true_env.agent_pos),
            "agent_dir": copy.deepcopy(true_env.agent_dir),
            "carrying": carrying_type
        }

    def get_progress(self) -> Dict[str, Any]:
        return {}

    def supports_simulation(self) -> bool:
        return False

    def supports_goal_distance(self) -> bool:
        return False

    def supports_stop(self) -> bool:
        return True

    def get_termination_message(self, success: bool) -> str:
        if success:
            return "Target object successfully picked up! Mission accomplished."
        return "Failed to pick up the target object. Mission failed."

    def validate_state_consistency(self):
        pass

def apply_env_wrappers(env):
    """Apply BabyAI-specific wrappers for the environment."""
    return env
