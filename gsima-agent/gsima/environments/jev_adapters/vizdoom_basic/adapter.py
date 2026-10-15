"""Jev Adapter for ViZDoom Basic environment."""
import logging
from typing import Any, Dict

import vizdoom
from gsima.environments.base import BaseAdapter
from gsima.environments.jev_adapters.vizdoom_basic.schema import SUPPORTED_ACTIONS, VizdoomAction

class VizdoomBasicAdapter(BaseAdapter):
    """Jev-specific Action adapter for ViZDoom Basic."""
    
    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        
        # We MUST bypass grid BFS completely and use vector steering.
        # To do this, we need the labels buffer enabled to extract coordinates.
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        if game is not None:
            # We intercept and re-initialize the game engine with labels buffer enabled.
            logging.info("Re-initializing ViZDoom engine to enable Labels Buffer for Jev vector steering.")
            game.close()
            game.set_labels_buffer_enabled(True)
            game.init()

    def _get_env_action(self, action: VizdoomAction):
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        button_map = getattr(getattr(self.env, "unwrapped", self.env), "button_map", None)

        if game is not None and button_map is not None:
            desired_button = {
                VizdoomAction.MOVE_LEFT: vizdoom.Button.MOVE_LEFT,
                VizdoomAction.MOVE_RIGHT: vizdoom.Button.MOVE_RIGHT,
                VizdoomAction.SHOOT: vizdoom.Button.ATTACK,
            }.get(action)

            if desired_button is not None:
                available_buttons = list(game.get_available_buttons())
                if desired_button in available_buttons:
                    button_index = available_buttons.index(desired_button)
                    for discrete_action, button_vector in enumerate(button_map):
                        if button_vector[button_index] == 1:
                            return discrete_action

        # Fallback for VizdoomBasic-v1's Discrete(4) wrapper:
        if action == VizdoomAction.SHOOT: return 1
        if action == VizdoomAction.MOVE_RIGHT: return 2
        if action == VizdoomAction.MOVE_LEFT: return 3
        if action == VizdoomAction.STOP: return -1
        raise ValueError(f"Unsupported action for ViZDoom: {action}")

    def translate_action(self, action_name: str) -> any:
        return self.action_map.get(action_name, self.action_map[VizdoomAction.STOP.name])

    def get_canonical_actions(self) -> list:
        return SUPPORTED_ACTIONS

    def get_state_summary(self) -> dict:
        info = {"ammo": 0, "health": 100}
        game = getattr(self.env.unwrapped, "game", None)
        if game is not None and hasattr(game, "get_game_variable"):
            try:
                info["ammo"] = game.get_game_variable(vizdoom.GameVariable.AMMO2)
                info["health"] = game.get_game_variable(vizdoom.GameVariable.HEALTH)
            except Exception:
                pass
        
        enemies_visible = False
        if game is not None:
            state = game.get_state()
            if state and state.labels:
                for label in state.labels:
                    name = label.object_name.lower()
                    if "cacodemon" in name or "hellknight" in name or "monster" in name or "demon" in name:
                        enemies_visible = True
                        break
        
        return {
            "ammo": info.get("ammo", 0),
            "health": info.get("health", 0),
            "enemies_visible": enemies_visible,
            "mechanics_hints": "If ammo is greater than 0 and enemy is visible, ALWAYS target 'shoot_enemy'. If no enemy is visible, target 'explore'."
        }

    def _locate_target(self, game):
        state = game.get_state()
        if state and state.labels:
            for label in state.labels:
                name = label.object_name.lower()
                if "cacodemon" in name or "hellknight" in name or "monster" in name or "demon" in name:
                    return label
        return None

    def pathfind(self, target_class: str) -> str:
        """
        Bypasses Grid BFS. Reads ViZDoom engine RAM labels to find enemy coordinates.
        Calculates angular vector and translates to continuous steering output.
        """
        game = getattr(self.env.unwrapped, "game", None)
        if not game:
            return "STOP"

        if "enemy" in target_class:
            enemy = self._locate_target(game)
            if enemy:
                # Get vector relative to player's center view
                screen_width = game.get_screen_width()
                center = screen_width / 2.0
                
                # Jev Action Router logic for vector steering:
                # If enemy angle (relative horizontal pixel) is left of center, MOVE_LEFT.
                # If right, MOVE_RIGHT.
                # If aligned (tolerance of 20 pixels), SHOOT.
                if enemy.x < center - 20:
                    return "MOVE_LEFT"
                elif enemy.x > center + 20:
                    return "MOVE_RIGHT"
                else:
                    return "SHOOT"
            else:
                return "MOVE_RIGHT" # Scan
                
        return "STOP"
