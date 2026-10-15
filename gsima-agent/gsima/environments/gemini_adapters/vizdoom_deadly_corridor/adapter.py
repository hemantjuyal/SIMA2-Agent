"""Adapter for ViZDoom Deadly Corridor Gymnasium environments (Gemini Runtime)."""
import logging
from typing import Any, Dict

import vizdoom
from gsima.environments.base import BaseAdapter
from gsima.environments.gemini_adapters.vizdoom_deadly_corridor.schema import (
    SUPPORTED_ACTIONS,
    VizdoomDeadlyCorridorAction,
)


class VizdoomDeadlyCorridorAdapter(BaseAdapter):
    """Adapter for ViZDoom Deadly Corridor environments."""

    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        logging.debug("VizdoomDeadlyCorridorAdapter initialized with action map: %s", self.action_map)

    def _get_env_action(self, action: VizdoomDeadlyCorridorAction) -> int:
        """Map canonical actions to a ViZDoom discrete action index."""
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        button_map = getattr(getattr(self.env, "unwrapped", self.env), "button_map", None)

        if game is not None and button_map is not None:
            desired_button = {
                VizdoomDeadlyCorridorAction.MOVE_LEFT: vizdoom.Button.MOVE_LEFT,
                VizdoomDeadlyCorridorAction.MOVE_RIGHT: vizdoom.Button.MOVE_RIGHT,
                VizdoomDeadlyCorridorAction.SHOOT: vizdoom.Button.ATTACK,
                VizdoomDeadlyCorridorAction.MOVE_FORWARD: vizdoom.Button.MOVE_FORWARD,
                VizdoomDeadlyCorridorAction.MOVE_BACKWARD: vizdoom.Button.MOVE_BACKWARD,
                VizdoomDeadlyCorridorAction.TURN_LEFT: vizdoom.Button.TURN_LEFT,
                VizdoomDeadlyCorridorAction.TURN_RIGHT: vizdoom.Button.TURN_RIGHT,
            }.get(action)

            if desired_button is not None:
                available_buttons = list(game.get_available_buttons())
                if desired_button in available_buttons:
                    button_index = available_buttons.index(desired_button)
                    for discrete_action, button_vector in enumerate(button_map):
                        if button_vector[button_index] == 1:
                            return discrete_action

        # Fallback discrete mapping
        if action == VizdoomDeadlyCorridorAction.STOP: return 0
        if action == VizdoomDeadlyCorridorAction.TURN_RIGHT: return 1
        if action == VizdoomDeadlyCorridorAction.TURN_LEFT: return 2
        if action == VizdoomDeadlyCorridorAction.MOVE_BACKWARD: return 3
        if action == VizdoomDeadlyCorridorAction.MOVE_FORWARD: return 4
        if action == VizdoomDeadlyCorridorAction.SHOOT: return 5
        if action == VizdoomDeadlyCorridorAction.MOVE_RIGHT: return 6
        if action == VizdoomDeadlyCorridorAction.MOVE_LEFT: return 7
        raise ValueError(f"Unsupported action for ViZDoom Deadly Corridor: {action}")

    def translate_action(self, action_name: str) -> any:
        return self.action_map.get(action_name, self.action_map[VizdoomDeadlyCorridorAction.STOP.name])

    def get_canonical_actions(self) -> list:
        return SUPPORTED_ACTIONS

    def get_outcome_from_reward(self, reward: float) -> str:
        if reward > 500:
            return "SUCCESS"
        elif reward < -50:
            return "DEATH"
        return "IN_PROGRESS"
