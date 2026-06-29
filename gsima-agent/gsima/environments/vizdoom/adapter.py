"""Adapter for ViZDoom Gymnasium environments."""
import logging
import re
from typing import Any, Dict

import vizdoom

from gsima.environments.base import BaseAdapter
from gsima.environments.vizdoom.schema import SUPPORTED_ACTIONS
from gsima.schema import CanonicalAction


class VizdoomAdapter(BaseAdapter):
    """Adapter for ViZDoom environments.

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
            CanonicalAction.MOVE_LEFT.name,
            CanonicalAction.MOVE_RIGHT.name,
            CanonicalAction.SHOOT.name,
        ]
        logging.debug("VizdoomAdapter initialized with action map: %s", self.action_map)

    def _get_env_action(self, action: CanonicalAction):
        """Map canonical actions to a ViZDoom discrete action index."""
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        button_map = getattr(getattr(self.env, "unwrapped", self.env), "button_map", None)

        if game is not None and button_map is not None:
            desired_button = {
                CanonicalAction.MOVE_LEFT: vizdoom.Button.MOVE_LEFT,
                CanonicalAction.MOVE_RIGHT: vizdoom.Button.MOVE_RIGHT,
                CanonicalAction.SHOOT: vizdoom.Button.ATTACK,
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
        if action == CanonicalAction.SHOOT:
            return 1
        if action == CanonicalAction.MOVE_RIGHT:
            return 2
        if action == CanonicalAction.MOVE_LEFT:
            return 3
        logging.error("Unsupported canonical action for ViZDoom: %s", action)
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
            return self.action_map[CanonicalAction.SHOOT.name]
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
        if re.search(r'\b(room|wall|door|corridor|hallway|maze|obstacle|fence)\b', raw, re.I):
            normalized['obstacles'] = 'environment layout is visible'

        return normalized

    def _parse_move_direction(self, structured_perception: Dict[str, Any]) -> str | None:
        """Return a canonical move action when the VLM reports a move direction."""
        direction = str(structured_perception.get("move_direction", "")).lower()
        if not direction:
            return None

        if "left" in direction:
            return CanonicalAction.MOVE_LEFT.name
        if "right" in direction:
            return CanonicalAction.MOVE_RIGHT.name
        if "forward" in direction:
            # VizDoomBasic-v1 does not expose explicit forward movement in the
            # canonical action set, so prefer a lateral reposition first.
            return CanonicalAction.MOVE_RIGHT.name

        return None

    def suggest_action(self, structured_perception: Dict[str, Any]) -> str | None:
        """Prefer move or shoot based on what the VLM actually says."""
        move_action = self._parse_move_direction(structured_perception)
        if move_action is not None:
            return move_action

        enemy_text = str(structured_perception.get("enemies", "")).lower()
        ammo_text = str(structured_perception.get("ammo", "")).lower()
        weapon_text = str(structured_perception.get("weapons", "")).lower()
        all_text = " ".join(str(value) for value in structured_perception.values()).lower()

        has_visible_enemy = any(
            keyword in enemy_text
            for keyword in (
                "enemy",
                "monster",
                "zombie",
                "target",
                "creature",
                "alien",
            )
        )
        if not has_visible_enemy:
            has_visible_enemy = any(
                keyword in all_text
                for keyword in (
                    "enemy",
                    "monster",
                    "zombie",
                    "hostile",
                    "target",
                    "creature",
                    "alien",
                )
            )

        no_ammo_reported = any(
            phrase in ammo_text
            for phrase in (
                "no ammo",
                "out of ammo",
                "empty",
                "none",
                "zero",
            )
        )
        has_ammo = any(
            keyword in ammo_text
            for keyword in (
                "ammo",
                "bullet",
                "round",
                "loaded",
                "shell",
                "clip",
                "plenty",
                "enough",
            )
        )
        if not has_ammo:
            try:
                state = self.get_current_env_state()
                has_ammo = any(
                    float(state.get(key, 0) or 0) > 0
                    for key in ("ammo", "selected_weapon_ammo")
                )
            except Exception:
                has_ammo = False

        has_weapon = any(
            keyword in weapon_text
            for keyword in (
                "gun",
                "weapon",
                "rifle",
                "pistol",
                "shotgun",
                "knife",
            )
        )
        if not has_weapon:
            has_weapon = any(
                keyword in all_text
                for keyword in ("gun", "weapon", "rifle", "pistol", "shotgun", "handgun")
            )

        if has_visible_enemy and not no_ammo_reported and (has_ammo or has_weapon):
            return CanonicalAction.SHOOT.name

        # If we have a weapon but no clear enemy, let the controller/planner decide.
        return None

    def get_fallback_action(self, structured_perception: Dict[str, Any]) -> str | None:
        """Use an adapter-owned search pattern when no model/planner action exists."""
        move_action = self._parse_move_direction(structured_perception)
        if move_action is not None:
            return move_action

        suggested = self.suggest_action(structured_perception)
        if suggested is not None:
            return suggested

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
