import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

class CustomMiniGridRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_pos = None
        self.target_pos = None
        self.last_dist_to_goal = None
        # Store the environment's specific action value for moving forward
        self.move_forward_action = self.unwrapped.actions.forward

    def _get_target_pos(self):
        """Helper to get the target (goal) position from the MiniGrid environment."""
        # MiniGrid environments typically have a 'goal_pos' attribute for the target
        # If not, we might need to search the grid for the 'goal' object type
        if hasattr(self.unwrapped, 'goal_pos') and self.unwrapped.goal_pos is not None:
            return np.array(self.unwrapped.goal_pos)
        
        # Fallback: search the grid for a 'goal' object
        # This is more general but potentially slower if called frequently
        for r in range(self.unwrapped.height):
            for c in range(self.unwrapped.width):
                cell = self.unwrapped.grid.get(c, r)
                if cell and cell.type == 'goal':
                    return np.array([c, r])
        
        # Default fallback for environments like Empty-Nursery-v0 where goal isn't explicit,
        # but the green square is usually at (width-2, height-2) in 'Empty' environments.
        grid_size = self.unwrapped.width
        return np.array([grid_size - 2, grid_size - 2])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        new_reward = -0.01 # 1. Per-step penalty

        current_pos = np.array(self.unwrapped.agent_pos)
        
        # Check if goal was reached in the current step (original reward > 0 and terminated)
        if terminated and reward > 0:
            new_reward += 10.0 # Large positive reward for reaching the goal
            return obs, new_reward, terminated, truncated, info # Return early if goal reached
        
        # Only proceed with distance and obstacle checks if episode is not done
        if not terminated and not truncated:

            # 2. Distance-based shaping reward
            if self.target_pos is not None:
                current_dist_to_goal = np.sum(np.abs(current_pos - self.target_pos))
                if self.last_dist_to_goal is not None:
                    if current_dist_to_goal < self.last_dist_to_goal:
                        new_reward += 0.05 # Moving closer
                    elif current_dist_to_goal > self.last_dist_to_goal:
                        new_reward -= 0.02 # Moving farther
                self.last_dist_to_goal = current_dist_to_goal
            
            # 3. Obstacle collision penalty
            front_cell_pos = self.unwrapped.front_pos
            front_cell = self.unwrapped.grid.get(*front_cell_pos)

            # Penalize only if the action was MOVE_FORWARD and position didn't change
            is_move_forward = (action == self.move_forward_action)
            collided_with_obstacle = (
                self.last_pos is not None and np.array_equal(current_pos, self.last_pos)
            )

            if is_move_forward and collided_with_obstacle:
                 # Check what kind of obstacle is in front
                 if front_cell is not None and (front_cell.type == 'wall' or front_cell.type == 'lava'):
                     new_reward -= 0.1 # Penalty for bumping into a wall or lava
            
        self.last_pos = current_pos
        return obs, new_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_pos = np.array(self.unwrapped.agent_pos)
        self.target_pos = self._get_target_pos()
        self.last_dist_to_goal = np.sum(np.abs(self.last_pos - self.target_pos)) if self.target_pos is not None else None
        return obs, info
