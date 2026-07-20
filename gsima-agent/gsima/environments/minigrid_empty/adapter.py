"""Adapter for Gym-MiniGrid environments."""
import logging
import copy
from typing import Dict, Any
import numpy as np
from minigrid.core.constants import DIR_TO_VEC

from gsima.environments.base import BaseAdapter
from gsima.environments.minigrid_empty.schema import SUPPORTED_ACTIONS, MiniGridAction

class MinigridEmptyAdapter(BaseAdapter):
    """Action adapter for Gym-MiniGrid Empty environments."""
    def __init__(self, env):
        super().__init__(env)
        # Dynamically build the action map from the supported actions schema
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        logging.debug(f"MinigridEmptyAdapter initialized with action map: {self.action_map}")
        # Store the last known environment state for validation
        self.last_known_state = None

    def _get_env_action(self, action: MiniGridAction):
        """Maps a local action to a specific MiniGrid environment action."""
        logging.debug(f"Mapping action '{action}' to environment action.")
        if action == MiniGridAction.MOVE_FORWARD:
            return self.env.unwrapped.actions.forward
        if action == MiniGridAction.TURN_LEFT:
            return self.env.unwrapped.actions.left
        if action == MiniGridAction.TURN_RIGHT:
            return self.env.unwrapped.actions.right
        if action == MiniGridAction.STOP:
            return -1 # Special sentinel value
        logging.error(f"Unsupported action for MiniGrid: {action}")
        raise ValueError(f"Unsupported action for MiniGrid: {action}")

    def translate_action(self, action_name: str) -> any:
        """Translates an action name for MiniGrid."""
        logging.debug(f"Translating action name '{action_name}' for MiniGrid.")
        action = self.action_map.get(action_name)
        if action is None:
            logging.warning(f"Unknown action '{action_name}' for MiniGrid. Defaulting to STOP.")
            return self.action_map[MiniGridAction.STOP.name]
        logging.debug(f"Translated '{action_name}' to environment action '{action}'.")
        return action

    def get_canonical_actions(self) -> list:
        """Returns a list of canonical actions supported by the MiniGrid environment."""
        return SUPPORTED_ACTIONS

    def get_env_metadata(self) -> Dict[str, Any]:
        """Describe MiniGrid-specific capabilities for the agent."""
        return {
            "env_type": "minigrid",
            "supports_distance": True,
            "supports_stop": True,
            "state_keys": ["agent_pos", "agent_dir"],
        }

    def get_action_metadata(self) -> Dict[str, Any]:
        """Expose action semantics that are safe for generic planning."""
        return {
            action.name: {
                "canonical": action.name,
                "supports_stop": action.name == "STOP",
            }
            for action in SUPPORTED_ACTIONS
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a compact, generic summary of the current MiniGrid state."""
        current_state = self.get_current_env_state()
        return {
            "agent_pos": current_state.get("agent_pos"),
            "agent_dir": current_state.get("agent_dir"),
        }

    def get_current_env_state(self) -> Dict[str, Any]:
        """Gets the current ground truth state from the environment."""
        true_env = self.env.unwrapped
        return {
            "agent_pos": np.array(true_env.agent_pos),
            "agent_dir": copy.deepcopy(true_env.agent_dir),
        }

    def get_progress(self) -> Dict[str, Any]:
        """Return an empty dict since we no longer use internal progress."""
        return {}

    def supports_simulation(self) -> bool:
        """MiniGrid supports deterministic trajectory simulation, but disabled for SIMA2."""
        return False

    def supports_goal_distance(self) -> bool:
        """Disabled for SIMA2."""
        return False

    def supports_stop(self) -> bool:
        """MiniGrid supports STOP as a meaningful terminal action."""
        return True

    def get_termination_message(self, success: bool) -> str:
        if success:
            return "Reached the green goal square! Mission accomplished."
        return "Failed to reach the green goal square. Mission failed."

    def validate_state_consistency(self):
        """
        Validates that the environment state is consistent.
        Logs warnings if there are discrepancies.
        """
        true_env = self.env.unwrapped
        current_pos = np.array(true_env.agent_pos)
        current_dir = true_env.agent_dir
        
        if self.last_known_state is not None:
            last_pos = self.last_known_state.get("agent_pos")
            last_dir = self.last_known_state.get("agent_dir")
            
            if last_pos is not None and not np.array_equal(last_pos, current_pos):
                logging.debug(f"State consistency check: Position changed from {last_pos} to {current_pos}")
            if last_dir is not None and last_dir != current_dir:
                logging.debug(f"State consistency check: Direction changed from {last_dir} to {current_dir}")
        
        self.last_known_state = {"agent_pos": current_pos, "agent_dir": current_dir}




def apply_env_wrappers(env):
    """Apply MiniGrid-specific wrappers for the environment."""
    return env


