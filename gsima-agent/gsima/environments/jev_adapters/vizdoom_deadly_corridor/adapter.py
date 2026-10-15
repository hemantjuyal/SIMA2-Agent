"""Jev Adapter for ViZDoom Deadly Corridor environment."""
import logging
from typing import Any, Dict, List, Optional

import vizdoom
from gsima.environments.base import BaseAdapter
from gsima.environments.jev_adapters.vizdoom_deadly_corridor.schema import (
    SUPPORTED_ACTIONS,
    VizdoomDeadlyCorridorAction,
)

ENEMY_TYPES = {
    "zombieman",
    "shotgunguy",
    "chaingunguy",
    "demon",
    "doomimp",
    "hellknight",
    "baronofhell",
    "cacodemon",
}

NON_ENEMIES = {
    "greenarmor",
    "doomplayer",
    "blood",
    "clip",
    "medikit",
    "stimpack",
    "bulletpuff",
    "shotgun",
    "chaingun",
    "chainsaw",
    "armor",
}


class VizdoomDeadlyCorridorAdapter(BaseAdapter):
    """Jev-specific Action adapter for ViZDoom Deadly Corridor."""

    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }

        self.consecutive_shots = 0

        # We need the labels buffer enabled to extract coordinates and entity metadata.
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        if game is not None:
            logging.info("Re-initializing ViZDoom engine to enable Labels Buffer for Jev Deadly Corridor.")
            game.close()
            game.set_labels_buffer_enabled(True)
            game.init()

    def _get_env_action(self, action: VizdoomDeadlyCorridorAction) -> int:
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

        # Fallback for standard VizdoomDeadlyCorridor-v1 Discrete(8) mapping
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

    def _get_living_enemies(self, game) -> List[Any]:
        state = game.get_state()
        if not state or not state.labels:
            return []
        living = []
        for label in state.labels:
            name = label.object_name.lower()
            if name.startswith("dead") or name in NON_ENEMIES:
                continue
            if any(t in name for t in ENEMY_TYPES):
                living.append(label)
        return living

    def _get_goal(self, game) -> Optional[Any]:
        state = game.get_state()
        if not state or not state.labels:
            return None
        for label in state.labels:
            if "armor" in label.object_name.lower():
                return label
        return None

    def get_state_summary(self) -> dict:
        health = 100.0
        ammo = 52.0
        game = getattr(self.env.unwrapped, "game", None)
        if game is not None and hasattr(game, "get_game_variable"):
            try:
                health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
                ammo = game.get_game_variable(vizdoom.GameVariable.AMMO2)
            except Exception:
                pass

        enemies = self._get_living_enemies(game) if game else []
        vest = self._get_goal(game) if game else None

        screen_width = game.get_screen_width() if game else 320
        center = screen_width / 2.0
        # An enemy is only an immediate threat if it has line of sight and full body profile (not occluded behind a wall)
        threatening = [
            e for e in enemies
            if (e.width >= 10 and e.height >= 20)
        ]

        if ammo <= 0:
            threat_count = 0
            hints = "AMMO EMPTY! You cannot shoot. You MUST target 'goal' to sprint forward to the green vest!"
        elif threatening:
            threat_count = len(threatening)
            hints = "Immediate hostile threat blocking advance in direct line of sight: target 'enemy' to eliminate."
        elif enemies:
            threat_count = 0
            hints = "Enemies detected nearby in corridor alcoves. Advance cautiously or engage emerging hostiles."
        else:
            threat_count = 0
            hints = "Corridor path is completely clear: target 'goal' to advance toward the green armor vest."

        return {
            "health": health,
            "ammo": ammo,
            "living_enemies_count": len(enemies),
            "immediate_threat_count": threat_count,
            "vest_visible": vest is not None,
            "mechanics_hints": hints,
        }

    def pathfind(self, target_class: str) -> str:
        """Dual-process System One pathfinder for Deadly Corridor.

        If target_class is 'enemy', aims at threatening monster and fires.
        If target_class is 'goal', centers on the green vest and moves forward.
        """
        game = getattr(self.env.unwrapped, "game", None)
        if not game:
            return "STOP"

        screen_width = game.get_screen_width()
        center = screen_width / 2.0

        ammo = 50.0
        if hasattr(game, "get_game_variable"):
            try:
                ammo = game.get_game_variable(vizdoom.GameVariable.AMMO2)
            except Exception:
                pass

        enemies = self._get_living_enemies(game)
        vest = self._get_goal(game)

        # If ammo is depleted, sprint straight towards vest
        if ammo <= 0:
            self.consecutive_shots = 0
            if vest:
                if vest.x < center - 20:
                    return "TURN_LEFT"
                elif vest.x > center + 20:
                    return "TURN_RIGHT"
            return "MOVE_FORWARD"

        visible_targets = [
            e for e in enemies
            if (e.width >= 10 and e.height >= 20)
        ]

        # PRIORITY REFLEX: If an unoccluded monster is visible with ammo available,
        # eliminate it regardless of whether System Two said 'enemy' or 'goal'
        if visible_targets and getattr(self, "consecutive_shots", 0) < 8:
            target = min(visible_targets, key=lambda e: abs(e.x - center))
            if target.x < center - 15:
                self.consecutive_shots = 0
                return "TURN_LEFT"
            elif target.x > center + 15:
                self.consecutive_shots = 0
                return "TURN_RIGHT"
            else:
                self.consecutive_shots += 1
                return "SHOOT"

        # ADVANCE: No visible threats directly in line of fire, advance towards vest
        self.consecutive_shots = 0
        if vest:
            if vest.x < center - 20:
                return "TURN_LEFT"
            elif vest.x > center + 20:
                return "TURN_RIGHT"
            else:
                return "MOVE_FORWARD"

        return "MOVE_FORWARD"

    def check_success(self, total_reward: float, player_alive: bool) -> bool:
        """In Deadly Corridor, true success requires surviving and securing the green vest."""
        return player_alive and total_reward > 800.0

    def get_termination_message(self, success: bool) -> str:
        if success:
            return "Mission accomplished! The green vest has been secured and the corridor cleared."
        return "Mission failed. The agent was eliminated by hostile fire before reaching the objective."
