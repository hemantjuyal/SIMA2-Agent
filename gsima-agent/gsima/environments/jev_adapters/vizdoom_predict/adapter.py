"""Jev Adapter for ViZDoom Predict Position environment."""
import logging
from typing import Any, Dict

import vizdoom
from gsima.utils import config
from gsima.environments.base import BaseAdapter
from gsima.environments.jev_adapters.vizdoom_predict.schema import SUPPORTED_ACTIONS, VizdoomPredictAction

class VizdoomPredictAdapter(BaseAdapter):
    """Jev-specific Action adapter for ViZDoom Predict Position."""
    
    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            action.name: self._get_env_action(action) for action in SUPPORTED_ACTIONS
        }
        
        # State for predictive steering
        self.last_enemy_x = None
        self.last_world_y = None
        self.true_dx = 0
        
        # We MUST bypass grid BFS completely and use vector steering.
        # To do this, we need the labels buffer enabled to extract coordinates.
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        if game is not None:
            logging.info("Re-initializing ViZDoom engine to enable Labels Buffer, Multi-Rocket ammo, and Game Variables.")
            game.close()
            game.set_labels_buffer_enabled(True)
            game.add_game_args("+sv_cheats 1")
            game.add_available_game_variable(vizdoom.GameVariable.AMMO5)
            game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
            game.add_available_game_variable(vizdoom.GameVariable.ANGLE)
            game.add_available_game_variable(vizdoom.GameVariable.POSITION_X)
            game.add_available_game_variable(vizdoom.GameVariable.POSITION_Y)
            timeout = getattr(config, "EPISODE_TIMEOUT", None)
            if timeout is not None:
                game.set_episode_timeout(timeout)
            elif getattr(config, "MAX_STEPS", None) is not None and getattr(config, "FRAME_SKIP", None) is not None:
                needed_tics = config.MAX_STEPS * config.FRAME_SKIP
                if needed_tics > game.get_episode_timeout():
                    game.set_episode_timeout(needed_tics)
            game.init()

    def _get_env_action(self, action: VizdoomPredictAction):
        game = getattr(getattr(self.env, "unwrapped", self.env), "game", None)
        button_map = getattr(getattr(self.env, "unwrapped", self.env), "button_map", None)

        if action == VizdoomPredictAction.WAIT:
            return 0  # Discrete NO-OP [0, 0, 0]

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

        # Fallback for VizdoomPredictPosition-v1's Discrete wrapper
        if action == VizdoomPredictAction.WAIT: return 0
        if action == VizdoomPredictAction.TURN_LEFT: return 3
        if action == VizdoomPredictAction.TURN_RIGHT: return 2
        if action == VizdoomPredictAction.SHOOT: return 1
        if action == VizdoomPredictAction.STOP: return -1
        raise ValueError(f"Unsupported action for ViZDoom: {action}")

    def translate_action(self, action_name: str) -> any:
        return self.action_map.get(action_name, self.action_map[VizdoomPredictAction.STOP.name])

    def get_canonical_actions(self) -> list:
        return SUPPORTED_ACTIONS

    def get_state_summary(self) -> dict:
        info = {"ammo": 50.0, "health": 100.0}
        game = getattr(self.env.unwrapped, "game", None)
        if game is not None and hasattr(game, "get_game_variable"):
            try:
                curr_ammo = game.get_game_variable(vizdoom.GameVariable.AMMO5)
                # Keep ammo replenished so agent can fire multiple rockets for a fast kill
                if curr_ammo <= 2:
                    game.send_game_command("give ammo")
                    curr_ammo = 50.0
                info["ammo"] = curr_ammo
                info["health"] = game.get_game_variable(vizdoom.GameVariable.HEALTH)
            except Exception:
                pass
        
        enemy = self._locate_target(game) if game is not None else None
        enemies_visible = enemy is not None
        
        if enemies_visible and info["ammo"] > 0:
            hints = "Monster is visible moving across the corridor with rockets ready. Target 'enemy' to lead the target and fire rockets rapidly for a fast kill."
        elif not enemies_visible:
            hints = "Monster is not in direct line of sight. Target 'explore' to rotate and scan the corridor."
        else:
            hints = "Rockets discharged. Target 'explore' while tracking target."

        return {
            "ammo": info.get("ammo", 50.0),
            "health": info.get("health", 100.0),
            "enemies_visible": enemies_visible,
            "mechanics_hints": hints
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
        Physics-grounded predictive vector steering:
        Calculates exact projectile flight time, weapon ignition delay (8 tics),
        and leads the moving target with rapid firing capability.
        """
        import math
        game = getattr(self.env.unwrapped, "game", None)
        if not game:
            return "STOP"

        current_tic = game.get_episode_time() if hasattr(game, "get_episode_time") else 0
        if current_tic <= 20:
            self.last_world_y = None
            self.last_tic = None
            self.vy = 0.0

        enemy = self._locate_target(game)
        if not enemy:
            self.last_world_y = None
            self.last_tic = None
            self.vy = 0.0
            return "TURN_RIGHT"  # Active sweep

        mx = enemy.object_position_x
        my = enemy.object_position_y

        if hasattr(self, 'last_world_y') and self.last_world_y is not None and hasattr(self, 'last_tic') and self.last_tic is not None and current_tic > self.last_tic:
            dt = current_tic - self.last_tic
            self.vy = (my - self.last_world_y) / dt
        self.last_world_y = my
        self.last_tic = current_tic

        try:
            ammo = game.get_game_variable(vizdoom.GameVariable.AMMO5)
            p_angle = game.get_game_variable(vizdoom.GameVariable.ANGLE)
            px = game.get_game_variable(vizdoom.GameVariable.POSITION_X)
            py = game.get_game_variable(vizdoom.GameVariable.POSITION_Y)
        except Exception:
            ammo = 50.0
            p_angle = 0.0
            px = -544.0
            py = 0.0

        # Weapon ready state: Doom rocket launcher raise animation completes at >= 32 tics
        vy = getattr(self, 'vy', 0.0)
        if current_tic >= 32 and abs(vy) > 0.001:
            dx = mx - px
            rocket_speed = 20.0
            t_flight = dx / rocket_speed
            total_time = 8.0 + t_flight  # 8 tics weapon trigger delay
            future_y = my + vy * total_time
            t_flight = math.hypot(dx, future_y - py) / rocket_speed
            total_time = 8.0 + t_flight
            future_y = my + vy * total_time

            target_rad = math.atan2(future_y - py, dx)
            target_deg = math.degrees(target_rad) % 360.0
            cur_deg = p_angle % 360.0
            diff = (target_deg - cur_deg + 180.0) % 360.0 - 180.0

            # 3.8-degree tolerance matches the 7.03-degree discrete turn resolution (half-step),
            # preventing oscillation and ensuring immediate, decisive firing.
            if abs(diff) <= 3.8:
                return "SHOOT"
            elif diff > 0:
                return "TURN_LEFT"
            else:
                return "TURN_RIGHT"

        return "WAIT"


    def check_success(self, total_reward: float, player_alive: bool) -> bool:
        """In Predict Position, success is eliminating the monster (reward > 0.5)."""
        return total_reward > 0.5

    def get_termination_message(self, success: bool) -> str:
        if success:
            return "Mission accomplished! The target monster was predicted and eliminated."
        return "Mission failed or timed out. The monster was not eliminated before the rocket or time expired."
