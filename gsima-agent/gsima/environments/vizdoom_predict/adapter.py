"""Adapter for ViZDoom Gymnasium environments."""
import logging
import re
from typing import Any, Dict

import vizdoom

from gsima.environments.base import BaseAdapter
from gsima.environments.vizdoom_predict.schema import SUPPORTED_ACTIONS, VizdoomPredictAction

class VizdoomPredictAdapter(BaseAdapter):
    """Adapter for ViZDoom Predict Position environments.

    The implementation keeps the environment-specific action translation logic
    isolated so the core agent can remain generic.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        self._combat_search_index = 0
        self._combat_search_pattern = [
            VizdoomPredictAction.TURN_LEFT.name,
            VizdoomPredictAction.TURN_RIGHT.name,
            VizdoomPredictAction.SHOOT.name,
        ]
        logging.debug("VizdoomPredictAdapter initialized with action map: %s", self.action_map)

    def _get_env_action(self, action: VizdoomPredictAction):
        """Map canonical actions to a ViZDoom discrete action index."""
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        button_map = getattr(getattr(self.env, "unwrapped", self.env), "button_map", None)

        if game is not None and button_map is not None:
            desired_button = {
                VizdoomPredictAction.TURN_LEFT: vizdoom.Button.TURN_LEFT,
                VizdoomPredictAction.TURN_RIGHT: vizdoom.Button.TURN_RIGHT,
                VizdoomPredictAction.SHOOT: vizdoom.Button.ATTACK,
            }.get(action)

            if desired_button is not None:
                available_buttons = list(game.get_available_buttons())
                if desired_button in available_buttons:
                    button_index = available_buttons.index(desired_button)
                    for discrete_action, button_vector in enumerate(button_map):
                        if button_vector[button_index] == 1:
                            return discrete_action

        # Fallback for VizdoomBasic-v1's Discrete(4) wrapper:
        # [NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT].
        if action == VizdoomPredictAction.TURN_LEFT:
            return 1
        if action == VizdoomPredictAction.TURN_RIGHT:
            return 2
        if action == VizdoomPredictAction.SHOOT:
            return 3
        if action == VizdoomPredictAction.STOP:
            return -1
        logging.error("Unsupported action for ViZDoom: %s", action)
        raise ValueError(f"Unsupported action for ViZDoom: {action}")

    def translate_action(self, action_name: str) -> Any:
        """Translate a canonical action name into a ViZDoom step action."""
        logging.debug("Translating action name '%s' for ViZDoom.", action_name)
        action = self.action_map.get(action_name)
        if action is None:
            logging.warning(
                "Unknown action '%s' for ViZDoom. Falling back to SHOOT.",
                action_name,
            )
            return self.action_map[VizdoomPredictAction.SHOOT.name]
        return action

    def get_canonical_actions(self) -> list:
        """Return the canonical actions supported by the ViZDoom environment."""
        return SUPPORTED_ACTIONS

    def get_env_metadata(self) -> Dict[str, Any]:
        return {
            "env_type": "vizdoom",
            "supports_distance": False,
            "supports_stop": False,
            "state_keys": [
                "ammo",
                "health",
                "selected_weapon_ammo",
                "frags",
                "enemies_visible",
            ],
        }

    def get_action_metadata(self) -> Dict[str, Any]:
        return {
            action.name: {
                "canonical": action.name,
                "supports_stop": False,
            }
            for action in SUPPORTED_ACTIONS
        }

    def normalize_perception(
        self,
        structured_perception: Dict[str, Any],
        raw_response: str = "",
    ) -> Dict[str, Any]:
        """Normalize VLM combat-scene labels for ViZDoom."""
        key_aliases = {
            "enemy": "enemies",
            "monster": "enemies",
            "target": "enemies",
            "weapon": "weapons",
            "obstacle": "obstacles",
            "action": "recommended_action",
            "recommended_action": "recommended_action",
            "aligned": "aligned",
            "aligned_for_shot": "aligned",
            "move_direction": "move_direction",
        }
        normalized = {
            key_aliases.get(key, key): value
            for key, value in structured_perception.items()
        }

        if normalized:
            return normalized

        raw = raw_response.strip()
        if re.search(r'\b(enemy|enemies|monster|zombie|target|creature|hostile)\b', raw, re.I):
            normalized['enemies'] = 'one or more hostile targets appear visible'
        if re.search(r'\b(weapon|gun|rifle|pistol|shotgun|handgun|aiming)\b', raw, re.I):
            normalized['weapons'] = 'a weapon is visible'
        if re.search(r'\b(ammo|bullet|round|clip|shell)\b', raw, re.I):
            normalized['ammo'] = 'ammo appears available'
        if re.search(r'\b(health|injury|hurt|damage|full health)\b', raw, re.I):
            normalized['health'] = 'player health status is mentioned'
        if re.search(r'\b(room|wall|door|corridor|narrow hallway|maze|obstacle|fence)\b', raw, re.I):
            normalized['obstacles'] = 'environment layout is visible'

        return normalized



    def get_fallback_action(self, structured_perception: Dict[str, Any]) -> str | None:
        """Use an adapter-owned search pattern when no model/planner action exists."""

        action = self._combat_search_pattern[
            self._combat_search_index % len(self._combat_search_pattern)
        ]
        self._combat_search_index += 1
        return action

    def should_explain_action(
        self,
        action_name: str,
        structured_perception: Dict[str, Any],
    ) -> bool:
        """Do not block fast combat reflexes on an explanation-only LLM call."""
        return False

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a lightweight snapshot of the environment state."""
        return self.get_current_env_state()

    def get_current_env_state(self) -> Dict[str, Any]:
        info = getattr(self.env, "info", {})
        game = None
        if hasattr(self.env, "unwrapped"):
            game = getattr(self.env.unwrapped, "game", None)
        if game is not None and hasattr(game, "get_game_variable"):
            info = {
                "ammo": game.get_game_variable(vizdoom.GameVariable.AMMO2),
                "health": game.get_game_variable(vizdoom.GameVariable.HEALTH),
                "selected_weapon_ammo": game.get_game_variable(
                    vizdoom.GameVariable.SELECTED_WEAPON_AMMO
                ),
                "frags": game.get_game_variable(vizdoom.GameVariable.FRAGCOUNT),
            }
        return info

    def supports_simulation(self) -> bool:
        return False

    def supports_goal_distance(self) -> bool:
        return False

    def supports_stop(self) -> bool:
        return False

    def get_termination_message(self, success: bool) -> str:
        if success:
            return "Target eliminated. Mission accomplished."
        return "Failed to eliminate the target. Mission failed."

    def validate_state_consistency(self) -> bool:
        return False
