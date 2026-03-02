"""Adapter for Gym-MiniGrid environments."""
import logging
import copy
from typing import Dict, Any
import numpy as np
from minigrid.core.constants import DIR_TO_VEC

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

    def simulate_next_state(self, action_name: str) -> Dict[str, Any]:
        """
        Simulates the next state for a given action using the ground truth from the environment.
        This provides a perfect, deterministic prediction.
        """
        logging.debug(f"--- SIMULATOR: Starting simulation for action '{action_name}' ---")
        true_env = self.env.unwrapped
        
        # Initial State Logging
        agent_pos = np.array(true_env.agent_pos)
        agent_dir = copy.deepcopy(true_env.agent_dir)
        logging.debug(f"SIMULATOR: Initial state -> pos: {agent_pos} (type: {type(agent_pos)}), dir: {agent_dir} (type: {type(agent_dir)})")

        grid = true_env.grid

        goal_pos = None
        for obj in grid.grid:
            if obj and obj.type == 'goal':
                goal_pos = obj.cur_pos
                break
        
        if goal_pos is None:
            raise RuntimeError("Could not find goal in MiniGrid environment.")

        goal_pos = np.array(goal_pos)

        # Simulate action effect
        if action_name == "TURN_LEFT":
            agent_dir = (agent_dir - 1) % 4
        elif action_name == "TURN_RIGHT":
            agent_dir = (agent_dir + 1) % 4
        elif action_name == "MOVE_FORWARD":
            fwd_pos = agent_pos + DIR_TO_VEC[agent_dir]
            fwd_cell = grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                agent_pos = fwd_pos
        
        logging.debug(f"SIMULATOR: State after action -> pos: {agent_pos}, dir: {agent_dir}")

        # Predict new perception from simulated state
        orientation_map = {0: 'EAST', 1: 'SOUTH', 2: 'WEST', 3: 'NORTH'}
        predicted_orientation = orientation_map[agent_dir]

        # Log before vector math
        dir_vec = DIR_TO_VEC[agent_dir]
        logging.debug(f"SIMULATOR: Debugging dir_vec access -> dir_vec: {dir_vec} (type: {type(dir_vec)}), agent_dir: {agent_dir} (type: {type(agent_dir)})")
        
        fwd_pos_after_move = agent_pos + dir_vec
        fwd_cell = grid.get(*fwd_pos_after_move)
        predicted_obstacle = 'true' if fwd_cell is not None and not fwd_cell.can_overlap() else 'false'

        if np.array_equal(agent_pos, goal_pos):
            goal_rel_pos = "here"
        else:
            vec_to_goal = goal_pos - agent_pos
            right_vec = np.array([dir_vec[1], -dir_vec[0]])
            
            logging.debug(f"SIMULATOR: Vector values -> vec_to_goal: {vec_to_goal}, dir_vec: {dir_vec}, right_vec: {right_vec}")

            dot_forward = np.dot(vec_to_goal, dir_vec)
            dot_right = np.dot(vec_to_goal, right_vec)

            if abs(dot_forward) > abs(dot_right):
                goal_rel_pos = 'front' if dot_forward > 0 else 'back'
            else:
                goal_rel_pos = 'right' if dot_right > 0 else 'left'
        
        result = {
            "agent_orientation": predicted_orientation,
            "goal_relative_position": goal_rel_pos,
            "obstacle_in_front": predicted_obstacle
        }
        logging.debug(f"--- SIMULATOR: Finished simulation. Result: {result} ---")
        return result

