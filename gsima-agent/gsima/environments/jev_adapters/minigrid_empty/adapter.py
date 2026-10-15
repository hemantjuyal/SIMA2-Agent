"""Jev Adapter for Gym-MiniGrid Empty environments."""
import logging
from collections import deque
from gsima.environments.base import BaseAdapter
from gsima.environments.gemini_adapters.minigrid_empty.schema import SUPPORTED_ACTIONS, MiniGridAction

class MinigridEmptyAdapter(BaseAdapter):
    """Jev-specific Action adapter for Gym-MiniGrid Empty."""
    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        logging.debug(f"Jev MinigridEmptyAdapter initialized.")

    def _get_env_action(self, action: MiniGridAction):
        if action == MiniGridAction.MOVE_FORWARD:
            return self.env.unwrapped.actions.forward
        if action == MiniGridAction.TURN_LEFT:
            return self.env.unwrapped.actions.left
        if action == MiniGridAction.TURN_RIGHT:
            return self.env.unwrapped.actions.right
        if action == MiniGridAction.STOP:
            return -1
        raise ValueError(f"Unsupported action: {action}")

    def translate_action(self, action_name: str) -> any:
        return self.action_map.get(action_name, self.action_map[MiniGridAction.STOP.name])

    def get_canonical_actions(self) -> list:
        return SUPPORTED_ACTIONS

    def _locate_target(self, target_class: str):
        env_grid = self.env.unwrapped.grid
        for x in range(env_grid.width):
            for y in range(env_grid.height):
                cell = env_grid.get(x, y)
                if cell is not None:
                    if target_class == "goal" and cell.type == "goal":
                        return (x, y)
        return None

    def pathfind(self, target_class: str) -> str:
        """
        Executes System One routing.
        Finds the target, computes the next BFS tile, and calculates the steering action.
        """
        target_coords = self._locate_target(target_class)
        if not target_coords:
            logging.warning(f"Target '{target_class}' not found on grid.")
            return "STOP"

        true_env = self.env.unwrapped
        start = tuple(true_env.agent_pos)
        goal = tuple(target_coords)

        if start == goal:
            return "STOP"

        # Local BFS
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
                        if cell is None or cell.type not in ["wall"] or neighbor == goal:
                            visited.add(neighbor)
                            new_path = list(path)
                            new_path.append(neighbor)
                            queue.append(new_path)

        if not next_tile:
            # Already adjacent or no path
            front_cell = true_env.front_pos
            if tuple(front_cell) == goal:
                return "MOVE_FORWARD"
            return "STOP"

        # Action Router (Vector Math)
        px, py = true_env.agent_pos
        nx, ny = next_tile
        current_dir = true_env.agent_dir  # 0: Right, 1: Down, 2: Left, 3: Up
        
        if nx > px: target_dir = 0
        elif nx < px: target_dir = 2
        elif ny > py: target_dir = 1
        elif ny < py: target_dir = 3
        else: return "MOVE_FORWARD" # Should not happen
        
        if current_dir == target_dir:
            return "MOVE_FORWARD"
        elif (target_dir - current_dir) % 4 == 1:
            return "TURN_RIGHT"
        else:
            return "TURN_LEFT"
