"""Adapter for BabyAI UnlockPickup environment."""
import logging
import copy
from typing import Dict, Any
import numpy as np

from gsima.environments.base import BaseAdapter
from gsima.environments.gemini_adapters.babyai_unlock_pickup.schema import SUPPORTED_ACTIONS, BabyAIAction

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
        true_env = self.env.unwrapped
        
        # Describe what is directly in front of the agent
        front_cell = true_env.grid.get(*true_env.front_pos)
        if front_cell is None:
            front_object = "empty space"
        else:
            if front_cell.type == "door":
                front_object = f"{front_cell.color} door ({'open' if front_cell.is_open else 'closed'})"
            else:
                front_object = f"{front_cell.color} {front_cell.type}"
                
        # Describe all objects and their relative directions
        objects_info = []
        ax, ay = current_state.get("agent_pos")
        adir = current_state.get("agent_dir") # 0: Right, 1: Down, 2: Left, 3: Up
        
        # Algorithm: Global Pathfinding Planner (BFS)
        # This provides a deterministic next-step vector for the System One Jev model,
        # solving local minima around walls without requiring the model to reason spatially.
        def get_next_bfs_step(start_x, start_y, target_x, target_y):
            from collections import deque
            queue = deque([(start_x, start_y, [])])
            visited = set([(start_x, start_y)])
            while queue:
                cx, cy, path = queue.popleft()
                if cx == target_x and cy == target_y:
                    if not path:
                        return 0, 0
                    return path[0][0] - start_x, path[0][1] - start_y
                for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                    if 0 <= nx < true_env.grid.width and 0 <= ny < true_env.grid.height:
                        if (nx, ny) not in visited:
                            c = true_env.grid.get(nx, ny)
                            if c is None or c.type != 'wall' or (nx == target_x and ny == target_y):
                                visited.add((nx, ny))
                                queue.append((nx, ny, path + [(nx, ny)]))
            return target_x - start_x, target_y - start_y

        for x in range(true_env.grid.width):
            for y in range(true_env.grid.height):
                cell = true_env.grid.get(x, y)
                if cell is not None and cell.type not in ["wall", "floor"]:
                    obj_desc = f"{cell.color} {cell.type}" if cell.type != "door" else f"{cell.color} door ({'open' if cell.is_open else 'closed'})"
                    
                    dx, dy = get_next_bfs_step(ax, ay, x, y)
                    
                    if dx == 0 and dy == 0:
                        continue
                    
                    if adir == 0:
                        forward, right = dx, dy
                    elif adir == 1:
                        forward, right = dy, -dx
                    elif adir == 2:
                        forward, right = -dx, -dy
                    elif adir == 3:
                        forward, right = -dy, dx
                        
                    rel_desc = []
                    if forward > 0: rel_desc.append("forward")
                    elif forward < 0: rel_desc.append("backward")
                    if right > 0: rel_desc.append("right")
                    elif right < 0: rel_desc.append("left")
                    
                    objects_info.append(f"{obj_desc} is {', '.join(rel_desc)}")
                    
        # Create an ASCII map of the agent's 7x7 field of view
        obs = self.env.unwrapped.gen_obs()
        image = obs['image']
        ascii_lines = []
        for j in range(7):
            line = []
            for i in range(7):
                if i == 3 and j == 6:
                    line.append("^") # Agent is always at bottom center, facing up
                    continue
                type_idx = image[i, j, 0]
                if type_idx == 1: line.append(".") # empty
                elif type_idx == 2: line.append("W") # wall
                elif type_idx == 4: line.append("D") # door
                elif type_idx == 5: line.append("K") # key
                elif type_idx == 8: line.append("G") # goal/box
                else: line.append("?")
            ascii_lines.append(" ".join(line))
        ascii_map = "\n".join(ascii_lines)
                
        return {
            "agent_pos": current_state.get("agent_pos"),
            "agent_dir": current_state.get("agent_dir"),
            "carrying": current_state.get("carrying"),
            "facing": front_object,
            "nearby_objects": objects_info,
            "visual_map_7x7": ascii_map
        }

    def get_current_env_state(self) -> Dict[str, Any]:
        true_env = self.env.unwrapped
        if true_env.carrying:
            carrying_str = f"{true_env.carrying.color} {true_env.carrying.type}"
        else:
            carrying_str = None
            
        return {
            "agent_pos": np.array(true_env.agent_pos),
            "agent_dir": copy.deepcopy(true_env.agent_dir),
            "carrying": carrying_str
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
