"""Jev Adapter for BabyAI UnlockPickup environment."""
import logging
from collections import deque
from gsima.environments.base import BaseAdapter
from gsima.environments.gemini_adapters.babyai_unlock_pickup.schema import SUPPORTED_ACTIONS, BabyAIAction

class BabyaiUnlockPickupAdapter(BaseAdapter):
    """Jev-specific Action adapter for BabyAI."""
    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }

    def _get_env_action(self, action: BabyAIAction):
        if action == BabyAIAction.MOVE_FORWARD: return self.env.unwrapped.actions.forward
        if action == BabyAIAction.TURN_LEFT: return self.env.unwrapped.actions.left
        if action == BabyAIAction.TURN_RIGHT: return self.env.unwrapped.actions.right
        if action == BabyAIAction.PICK_UP: return self.env.unwrapped.actions.pickup
        if action == BabyAIAction.DROP: return self.env.unwrapped.actions.drop
        if action == BabyAIAction.TOGGLE: return self.env.unwrapped.actions.toggle
        if action == BabyAIAction.STOP: return -1
        raise ValueError(f"Unsupported action: {action}")

    def translate_action(self, action_name: str) -> any:
        return self.action_map.get(action_name, self.action_map[BabyAIAction.STOP.name])

    def get_canonical_actions(self) -> list:
        return SUPPORTED_ACTIONS

    def get_state_summary(self) -> dict:
        true_env = self.env.unwrapped
        
        objects_seen = set()
        for x in range(true_env.grid.width):
            for y in range(true_env.grid.height):
                cell = true_env.grid.get(x, y)
                if cell is not None and cell.type not in ["wall", "floor"]:
                    if cell.type == "door":
                        objects_seen.add(f"{cell.color} door ({'open' if cell.is_open else 'closed'})")
                    else:
                        objects_seen.add(f"{cell.color} {cell.type}")
        
        front_cell = true_env.grid.get(*true_env.front_pos)
        front_desc = "empty"
        if front_cell:
            if front_cell.type == "door":
                front_desc = f"door ({'open' if front_cell.is_open else 'closed'})"
            else:
                front_desc = front_cell.type

        return {
            "objects_in_environment": list(objects_seen),
            "facing": front_desc,
            "mechanics_hints": "If the door is closed and you have no key, target 'key'. If the door is closed and you have the key, target 'door'. If the door is open, ALWAYS target 'box', regardless of your inventory. Do not target the key once the door is open."
        }

    def _locate_target(self, target_class: str):
        true_env = self.env.unwrapped
        for x in range(true_env.grid.width):
            for y in range(true_env.grid.height):
                cell = true_env.grid.get(x, y)
                if cell is not None:
                    # Match target_class string conceptually
                    if target_class in cell.type:
                        return (x, y)
        return None

    def pathfind(self, target_class: str) -> str:
        true_env = self.env.unwrapped
        target_coords = self._locate_target(target_class)
        
        if not target_coords:
            logging.warning(f"Target '{target_class}' not found on grid.")
            return "STOP"

        start = tuple(true_env.agent_pos)
        goal = tuple(target_coords)

        front_cell = true_env.grid.get(*true_env.front_pos)
        
        # Interaction Logic (System One Reflexes)
        if tuple(true_env.front_pos) == goal:
            if target_class == "door":
                if front_cell and front_cell.type == "door" and not front_cell.is_open:
                    return "TOGGLE"
                elif front_cell and front_cell.type == "door" and front_cell.is_open:
                    return "MOVE_FORWARD"
            elif target_class in ["box", "key"]:
                if true_env.carrying:
                    return "DROP"
                else:
                    return "PICK_UP"

        # Special drop logic if we are holding something we don't need
        if true_env.carrying and target_class not in ["door", true_env.carrying.type]:
            # If we need to interact with a box but we are holding a key, drop it in empty space!
            if not front_cell or front_cell.type == "floor":
                return "DROP"
            else:
                return "TURN_LEFT"

        if start == goal:
            return "STOP"

        # Local BFS for navigation
        queue = deque([[start]])
        visited = {start}
        next_tile = None
        
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            
            if (x, y) == goal:
                next_tile = path[1] if len(path) > 1 else path[0]
                break
                
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if 0 <= nx < true_env.grid.width and 0 <= ny < true_env.grid.height:
                    if neighbor not in visited:
                        cell = true_env.grid.get(nx, ny)
                        # In BabyAI, we can path through open doors!
                        can_pass = cell is None or cell.type == "floor" or neighbor == goal
                        if cell and cell.type == "door" and cell.is_open:
                            can_pass = True
                            
                        if can_pass:
                            visited.add(neighbor)
                            new_path = list(path)
                            new_path.append(neighbor)
                            queue.append(new_path)

        if not next_tile:
            # Maybe adjacent but blocked
            return "TURN_LEFT"

        # Action Router (Vector Math)
        px, py = true_env.agent_pos
        nx, ny = next_tile
        current_dir = true_env.agent_dir  # 0: Right, 1: Down, 2: Left, 3: Up
        
        if nx > px: target_dir = 0
        elif nx < px: target_dir = 2
        elif ny > py: target_dir = 1
        elif ny < py: target_dir = 3
        else: return "MOVE_FORWARD" 
        
        if current_dir == target_dir:
            return "MOVE_FORWARD"
        elif (target_dir - current_dir) % 4 == 1:
            return "TURN_RIGHT"
        else:
            return "TURN_LEFT"
