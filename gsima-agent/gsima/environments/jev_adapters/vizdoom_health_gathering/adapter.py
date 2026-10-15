"""Jev Adapter for ViZDoom Health Gathering environment."""
import logging
from typing import Any, Dict, List, Optional

import vizdoom
from gsima.environments.base import BaseAdapter
from gsima.environments.jev_adapters.vizdoom_health_gathering.schema import (
    SUPPORTED_ACTIONS,
    VizdoomHealthGatheringAction,
)


class VizdoomHealthGatheringAdapter(BaseAdapter):
    """Jev-specific Action adapter for ViZDoom Health Gathering."""

    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }

        # Target commitment tracking to prevent steering oscillation between multiple medikits
        self.committed_target_id: Optional[int] = None
        self.explore_step_count: int = 0

        # We need the labels buffer enabled to extract coordinates and entity metadata.
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        if game is not None:
            logging.info("Re-initializing ViZDoom engine to enable Labels Buffer for Jev Health Gathering.")
            game.close()
            game.set_labels_buffer_enabled(True)
            game.init()

    def _get_env_action(self, action: VizdoomHealthGatheringAction) -> int:
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        button_map = getattr(getattr(self.env, "unwrapped", self.env), "button_map", None)

        if game is not None and button_map is not None:
            desired_button = {
                VizdoomHealthGatheringAction.MOVE_FORWARD: vizdoom.Button.MOVE_FORWARD,
                VizdoomHealthGatheringAction.TURN_LEFT: vizdoom.Button.TURN_LEFT,
                VizdoomHealthGatheringAction.TURN_RIGHT: vizdoom.Button.TURN_RIGHT,
            }.get(action)

            if desired_button is not None:
                available_buttons = list(game.get_available_buttons())
                if desired_button in available_buttons:
                    button_index = available_buttons.index(desired_button)
                    for discrete_action, button_vector in enumerate(button_map):
                        if button_vector[button_index] == 1:
                            return discrete_action

        # Fallback for standard VizdoomHealthGathering-v1 Discrete(4) mapping
        if action == VizdoomHealthGatheringAction.STOP:
            return 0
        if action == VizdoomHealthGatheringAction.MOVE_FORWARD:
            return 1
        if action == VizdoomHealthGatheringAction.TURN_RIGHT:
            return 2
        if action == VizdoomHealthGatheringAction.TURN_LEFT:
            return 3
        raise ValueError(f"Unsupported action for ViZDoom Health Gathering: {action}")

    def translate_action(self, action_name: str) -> any:
        return self.action_map.get(action_name, self.action_map[VizdoomHealthGatheringAction.STOP.name])

    def get_canonical_actions(self) -> list:
        return SUPPORTED_ACTIONS

    def _get_visible_medikits(self, game) -> List[Any]:
        state = game.get_state()
        if not state or not state.labels:
            return []
        medikits = []
        for label in state.labels:
            name = label.object_name.lower()
            # In regular Health Gathering items are 'Medikit'; in Supreme they are 'CustomMedikit'.
            # Avoid toxic vials or poison objects in supreme scenarios.
            if "medikit" in name and "poison" not in name and "vial" not in name:
                medikits.append(label)
        return medikits

    def get_state_summary(self) -> dict:
        health = 100.0
        game = getattr(self.env.unwrapped, "game", None)
        if game is not None and hasattr(game, "get_game_variable"):
            try:
                health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
            except Exception:
                pass

        medikits = self._get_visible_medikits(game) if game else []

        if not medikits:
            hints = "No medikits currently in forward view. Scan and explore the chamber to find health packs."
        elif health < 40.0:
            hints = "CRITICAL HEALTH! Floor acid is damaging player rapidly. Must collect closest medikit immediately."
        else:
            hints = f"{len(medikits)} medikits visible. Navigate toward health packs to maintain maximum health."

        return {
            "health": health,
            "visible_medikits_count": len(medikits),
            "has_committed_target": self.committed_target_id is not None,
            "mechanics_hints": hints,
        }

    def pathfind(self, target_class: str) -> str:
        """Dual-process System One pathfinder for Health Gathering.

        Executes human-like navigation with target commitment to prevent
        angular oscillation between competing health packs across the room.
        """
        game = getattr(self.env.unwrapped, "game", None)
        if not game:
            return "STOP"

        screen_width = game.get_screen_width()
        center = screen_width / 2.0

        medikits = self._get_visible_medikits(game)

        # 1. Resolve target via commitment
        target = None
        if self.committed_target_id is not None:
            for m in medikits:
                if m.object_id == self.committed_target_id:
                    target = m
                    break

        # If committed target was collected or moved out of view, re-acquire
        if target is None:
            if medikits:
                # Prioritize largest medikit (closest in 3D perspective)
                target = max(medikits, key=lambda m: (m.width * m.height, -abs(m.x - center)))
                self.committed_target_id = target.object_id
            else:
                self.committed_target_id = None

        # 2. If target is active, steer toward it with anti-oscillation deadband (±30 px)
        if target is not None:
            self.explore_step_count = 0
            if target.x < center - 30:
                return "TURN_LEFT"
            elif target.x > center + 30:
                return "TURN_RIGHT"
            else:
                return "MOVE_FORWARD"

        # 3. Explore mode when no medikits are in line of sight: scan chamber
        self.committed_target_id = None
        self.explore_step_count += 1
        # Alternate scanning turns with occasional forward stride
        if self.explore_step_count % 5 == 0:
            return "MOVE_FORWARD"
        return "TURN_RIGHT"

    def check_success(self, total_reward: float, player_alive: bool) -> bool:
        """In Health Gathering, true success is surviving the acidic floor hazard."""
        return player_alive

    def get_termination_message(self, success: bool) -> str:
        if success:
            return "Mission accomplished! The agent successfully gathered health packs and survived the hazardous environment."
        return "Mission failed. The agent succumbed to the acidic floor before completing the survival objective."
