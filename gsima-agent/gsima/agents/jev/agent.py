import json
import logging
import os
import numpy as np
from typing import Any, Dict

from gsima.utils import config
from gsima.agents.base import BaseAgent
from gsima.agents.context import AgentContext
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

# Suppress noisy HTTP and SDK logging that dumps full headers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("typesafe_sdk").setLevel(logging.WARNING)

class JevAgent(BaseAgent):
    """
    An S1-SIMA Closed-Loop Agent using TypeSafe Jev as the System One Decision Model
    and Deterministic Software (Action Execution Layer) for kinematic and motor actuation.
    """
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        api_key = getattr(config, "TYPESAFE_API_KEY", "")
        if not api_key:
            logging.warning("TYPESAFE_API_KEY is missing. Jev agent will fail.")

        base_url = getattr(config, "TYPESAFE_BASE_URL", None) or os.getenv("TYPESAFE_BASE_URL", None)
        if not base_url:
            if api_key.startswith("sk-or-"):
                base_url = "https://openrouter.ai/api"
            else:
                base_url = "https://api.typesafe.ai"

        self.client = TypeSafeClient(
            api_key=api_key, 
            base_url=base_url
        )
        logging.info(f"Initialized S1-SIMA JevAgent '{name}' with System One + Action Execution Layer architecture (Base URL: {base_url}).")

    def _prepare_state_payload(self, context: AgentContext, initial_obs: Any) -> Dict[str, Any]:
        """Serializes clean state variables without generating heavy, error-prone text map blobs."""
        logging.debug("Preparing state payload and truncating memory.")
        
        is_carrying = context.env.unwrapped.carrying is not None if hasattr(context.env.unwrapped, "carrying") else False
        carrying_type = context.env.unwrapped.carrying.type if is_carrying else "None"
        
        mission_str = getattr(context.env.unwrapped, "mission", None)
        if not mission_str:
            mission_str = getattr(config, "INSTRUCTION", "Find the goal")

        raw_payload = {
            "mission": mission_str,
            "inventory_carrying": carrying_type,
            "environment": context.adapter.get_state_summary()
        }
        return raw_payload

    def run(self, context: AgentContext):
        """The main execution loop for the Jev agent."""
        if hasattr(config, "ENV_SEED") and config.ENV_SEED is not None:
            initial_obs, info = context.env.reset(seed=config.ENV_SEED)
        else:
            initial_obs, info = context.env.reset()
            
        context.memory_system.clear()
        logging.info("Environment reset and memory cleared.")

        # Obtain the RGB array strictly for UI/recording
        if config.RENDER_MODE in ("human", "record"):
            rgb_array_observation = context.env.unwrapped.render()
        else:
            rgb_array_observation = context.env.render()
            
        if context.event_emitter:
            context.event_emitter("frame", rgb_array_observation)

        terminated, truncated, step_count = False, False, 0
        total_reward = 0.0
        allowed_actions = {action.name for action in context.adapter.get_canonical_actions()}

        while not terminated and not truncated and step_count < config.MAX_STEPS:
            if config.RENDER_MODE == "human":
                context.env.render()
                
            logging.info(f"--- Step {step_count + 1}/{config.MAX_STEPS} ---")

            is_delay_phase = info.get("is_delay_phase", False)
            
            if is_delay_phase:
                import time
                time.sleep(0.05)
                action_name = "NOOP"
            else:
                try:
                    payload = self._prepare_state_payload(context, initial_obs)
                    
                    # Step 1: System One Decision Model (Jev Multi-Primitive Strategic Gating)
                    env_summary = payload.get("environment", {})
                    immediate_threat_count = env_summary.get("immediate_threat_count", 0)
                    ammo = env_summary.get("ammo", 50.0)

                    env_type = getattr(config, "ENV_TYPE", "").lower()
                    env_name = getattr(config, "GYM_ENVIRONMENT", "").lower()
                    is_vizdoom_deadly = "deadly" in env_type or "deadly" in env_name
                    is_vizdoom_health = "health" in env_type or "health" in env_name
                    is_vizdoom_predict = "predict" in env_type or "predict" in env_name
                    is_vizdoom_basic = "basic" in env_type or "basic" in env_name
                    is_babyai = "babyai" in env_type or "unlockpickup" in env_name or "babyai" in env_name
                    is_minigrid = "minigrid" in env_type or "empty" in env_name or "minigrid" in env_name

                    # Define TypeSafe primitives questions for each game environment
                    if is_vizdoom_deadly:
                        questions = {
                            "target": Choice(
                                instructions="Given the mission and threat environment, should the agent eliminate the immediate enemy threat or advance to the green vest goal?",
                                criteria={
                                    "enemy": "An immediate threat is visible in direct line of fire with ammo available; engage and eliminate it to survive.",
                                    "goal": "Immediate path is clear or ammo is depleted; advance towards the green armor vest objective."
                                }
                            ),
                            "is_clear_line_of_fire": Noul(
                                instructions="Does the agent have an unoccluded line of fire with sufficient ammunition to engage enemies?"
                            ),
                            "threat_level": Score(
                                instructions="Rate the lethal threat posed by visible hostiles in the corridor to the player.",
                                criteria=[
                                    "No enemies visible or corridor completely clear",
                                    "Distant or unaware enemy at long range",
                                    "Hostile closing distance, low immediate danger",
                                    "Active enemy in corridor with line of fire",
                                    "Multiple hostiles or imminent lethal attack requiring emergency suppression"
                                ]
                            )
                        }
                    elif is_vizdoom_health:
                        questions = {
                            "target": Choice(
                                instructions="Given the player's current health and visible health packs on the acidic floor, should the agent gather a medikit to restore health or explore and scan the room for items?",
                                criteria={
                                    "medikit": "A medikit is visible in the chamber; move directly to collect it and recover lost health.",
                                    "explore": "No medikits are directly in view; scan and explore the surrounding chamber to locate new health packs."
                                }
                            ),
                            "emergency_heal": Noul(
                                instructions="Is the agent in critical danger from floor acid requiring emergency medikit collection?"
                            ),
                            "medikit_priority": Score(
                                instructions="Rate the urgency of collecting health packs given current player health and acid hazard.",
                                criteria=[
                                    "Health is full (80-100%); explore or collect if convenient",
                                    "Health is moderate (50-79%); steady medikit collection recommended",
                                    "Health is degraded (30-49%); medikit collection prioritized over exploration",
                                    "Health is critical (<30%); emergency recovery mandatory to prevent death"
                                ]
                            )
                        }
                    elif is_vizdoom_predict:
                        questions = {
                            "target": Choice(
                                instructions="Given the moving monster across the corridor and your rocket launcher, should the agent aim and shoot at the enemy or scan to acquire target?",
                                criteria={
                                    "enemy": "A moving monster is detected; track its position, calculate lead, and fire.",
                                    "explore": "Target is not currently in line of sight; scan to acquire target."
                                }
                            ),
                            "is_aligned": Noul(
                                instructions="Is the weapon crosshair within the angular lead tolerance window to score a direct rocket hit?"
                            ),
                            "lead_error_severity": Score(
                                instructions="Rate the angular lead error between weapon crosshair and predicted target position.",
                                criteria=[
                                    "Target aligned within lead tolerance window; fire rocket",
                                    "Small angular offset; minor rotational correction needed",
                                    "Large angular offset; sweeping turn required to intercept target"
                                ]
                            )
                        }
                    elif is_vizdoom_basic:
                        questions = {
                            "target": Choice(
                                instructions="Given the monster on the far wall and your weapon, should the agent target and eliminate the enemy or reposition to acquire it?",
                                criteria={
                                    "enemy": "A monster is visible; align horizontally and fire to eliminate it.",
                                    "explore": "Target not aligned; reposition horizontally."
                                }
                            ),
                            "is_centered": Noul(
                                instructions="Is the monster aligned within the weapon crosshair deadband ready to be eliminated?"
                            )
                        }
                    elif is_babyai:
                        questions = {
                            "target": Choice(
                                instructions="Given the mission rules and current carrying inventory, which object class must be targeted next?",
                                criteria={
                                    "key": "The door is closed and the agent has no key; locate and retrieve the key.",
                                    "door": "The door is closed and the agent is holding the key; move to unlock and open the door.",
                                    "box": "The door is open; proceed directly to pickup the target box."
                                }
                            ),
                            "pickup_valid": Noul(
                                instructions="Is the agent directly facing the target item and ready to execute an interaction action?"
                            )
                        }
                    else:
                        # MiniGrid or generic default
                        questions = {
                            "target": Choice(
                                instructions="Given the agent position and orientation relative to the goal tile, which navigation target should be pursued?",
                                criteria={
                                    "goal": "Advance directly along clear grid path to step onto the green goal tile.",
                                    "explore": "Facing a boundary wall or obstruction; rotate to align with open corridor."
                                }
                            ),
                            "goal_adjacent": Noul(
                                instructions="Is the agent directly adjacent to the goal tile one step away from completion?"
                            )
                        }

                    # Prepare and log Jev Decisioning Request Payload (Formatted wire JSON matching OpenAPI specs)
                    jev_request_payload = {
                        "model": "jev-1.13",
                        "state": payload,
                        "questions": {k: q.model_dump() for k, q in questions.items()}
                    }
                    logging.info(f"Jev Decision Request Payload:\n{json.dumps(jev_request_payload, indent=2, default=str)}")

                    # Call Jev Backend via TypeSafe SDK (Speculative Fan-Out evaluating all primitives in parallel)
                    choice_confidence = 1.0
                    nouls_dict = {}
                    scores_dict = {}
                    try:
                        target_response = self.client.system_one(
                            model="jev-1.13",
                            state=payload,
                            questions=questions
                        )
                        jev_response_payload = (
                            target_response.model_dump(mode="json")
                            if hasattr(target_response, "model_dump")
                            else getattr(target_response, "__dict__", str(target_response))
                        )
                        logging.info(f"Jev Decision Response Payload:\n{json.dumps(jev_response_payload, indent=2, default=str)}")

                        # 1. Parse Choice Primitive
                        choice_ans = target_response.choices.get("target") or target_response.answers.get("target")
                        if choice_ans and hasattr(choice_ans, "choice"):
                            strategic_goal = choice_ans.choice
                            choice_confidence = getattr(choice_ans, "confidence", 1.0)
                        else:
                            strategic_goal = "goal"
                            choice_confidence = 0.5

                        # 2. Parse Noul Primitives (.noul is calibrated float in [0.0, 1.0])
                        for name, ans in target_response.nouls.items():
                            if hasattr(ans, "noul"):
                                nouls_dict[name] = ans.noul
                        for name, ans in target_response.answers.items():
                            if hasattr(ans, "noul") and name not in nouls_dict:
                                nouls_dict[name] = ans.noul

                        # 3. Parse Score Primitives (.score is continuous float, with legend & probabilities)
                        for name, ans in target_response.scores.items():
                            if hasattr(ans, "score"):
                                scores_dict[name] = {
                                    "score": ans.score,
                                    "confidence": getattr(ans, "confidence", None),
                                    "legend": getattr(ans, "legend", {}),
                                    "probabilities": getattr(ans, "probabilities", {})
                                }
                        for name, ans in target_response.answers.items():
                            if hasattr(ans, "score") and name not in scores_dict:
                                scores_dict[name] = {
                                    "score": ans.score,
                                    "confidence": getattr(ans, "confidence", None),
                                    "legend": getattr(ans, "legend", {}),
                                    "probabilities": getattr(ans, "probabilities", {})
                                }

                    except Exception as e:
                        logging.error(f"TypeSafe SDK failed: {e}")
                        if is_vizdoom_deadly:
                            strategic_goal = "enemy" if (immediate_threat_count > 0 and ammo > 0) else "goal"
                        elif is_vizdoom_health:
                            visible_medikits = env_summary.get("visible_medikits_count", 0)
                            strategic_goal = "medikit" if visible_medikits > 0 else "explore"
                        elif is_vizdoom_predict or is_vizdoom_basic:
                            strategic_goal = "enemy"
                        elif is_babyai:
                            inventory = payload.get("inventory_carrying", "None")
                            strategic_goal = "door" if inventory == "key" else "key"
                        else:
                            strategic_goal = "goal"
                        choice_confidence = 0.5
                        logging.info(f"Jev Fallback Decision Choice: {strategic_goal} (fallback due to error)")

                    # Multi-Primitive Tactical Integration & Guardrails
                    if is_vizdoom_health:
                        emergency_noul = nouls_dict.get("emergency_heal", 0.0)
                        medikit_score = scores_dict.get("medikit_priority", {}).get("score", 0.0)
                        visible_medikits = env_summary.get("visible_medikits_count", 0)
                        # Hazard reflex: trigger emergency medikit latching
                        if (emergency_noul > 0.8 or medikit_score >= 3.0) and visible_medikits > 0:
                            strategic_goal = "medikit"

                    elif is_vizdoom_deadly:
                        clear_fire_noul = nouls_dict.get("is_clear_line_of_fire", 1.0)
                        threat_score = scores_dict.get("threat_level", {}).get("score", 0.0)
                        # Tactical commitment: suppress lethal threat if ammo is available and line of fire is open
                        if immediate_threat_count > 0 and ammo > 0 and clear_fire_noul > 0.3:
                            strategic_goal = "enemy"
                        elif ammo <= 0 or (clear_fire_noul < 0.2 and immediate_threat_count == 0):
                            strategic_goal = "goal"

                    elif is_vizdoom_predict:
                        if env_summary.get("enemies_visible", False) and ammo > 0:
                            strategic_goal = "enemy"

                    elif is_babyai:
                        objects_seen = env_summary.get("objects_in_environment", [])
                        if any("door (open)" in str(o) for o in objects_seen):
                            strategic_goal = "box"

                    logging.info(f"Jev Strategic Gating output choice: {strategic_goal} (confidence: {choice_confidence:.2f})")

                    # Step 2: Action Execution Layer / Deterministic Software (Kinematics & Control)
                    action_name = context.adapter.pathfind(strategic_goal)

                    # Build telemetry narrative including Noul and Score primitive states
                    primitive_summaries = []
                    if nouls_dict:
                        primitive_summaries.append("Noul[" + ", ".join(f"{k}={v:.2f}" for k, v in nouls_dict.items()) + "]")
                    if scores_dict:
                        primitive_summaries.append("Score[" + ", ".join(f"{k}={v['score']:.2f}" for k, v in scores_dict.items()) + "]")
                    metrics_str = f" | {' '.join(primitive_summaries)}" if primitive_summaries else ""

                    if is_vizdoom_predict:
                        jev_thought = f"System One (Jev) targeted '{strategic_goal}' (conf: {choice_confidence:.2f}){metrics_str}. Action Execution Layer executed Predictive Vector Steering via Deterministic Software."
                    else:
                        jev_thought = f"System One (Jev) targeted '{strategic_goal}' (conf: {choice_confidence:.2f}){metrics_str}. Action Execution Layer routed trajectory via Deterministic Software."

                    logging.info(f"Jev Native Choice: {action_name} | Thought: {jev_thought}")
                except Exception as e:
                    logging.error(f"Execution failed: {e}")
                    action_name = next(iter(allowed_actions))
                    jev_thought = "Fallback random action due to failure."

            if context.event_emitter and not is_delay_phase:
                if is_vizdoom_health:
                    action_fallbacks = {
                        "MOVE_FORWARD": "I am moving forward to collect a medikit.",
                        "TURN_LEFT": "I am turning left to align with a medikit.",
                        "TURN_RIGHT": "I am turning right to scan for medikits.",
                        "STOP": "I have survived the hazardous environment."
                    }
                elif is_vizdoom_predict:
                    action_fallbacks = {
                        "MOVE_FORWARD": "I am moving forward.",
                        "TURN_LEFT": "I am turning left to lead the target.",
                        "TURN_RIGHT": "I am turning right to track the target.",
                        "SHOOT": "I am firing a rocket at the predicted position.",
                        "WAIT": "I am holding aim steady for target interception.",
                        "STOP": "I am standing by."
                    }
                elif is_vizdoom_basic:
                    action_fallbacks = {
                        "MOVE_LEFT": "I am strafing left.",
                        "MOVE_RIGHT": "I am strafing right.",
                        "SHOOT": "I am firing at the monster.",
                        "STOP": "I have eliminated the monster."
                    }
                else:
                    action_fallbacks = {
                        "MOVE_FORWARD": "I am moving forward.",
                        "TURN_LEFT": "I am turning left.",
                        "TURN_RIGHT": "I am turning right.",
                        "STOP": "I have reached the goal."
                    }
                narrative_str = action_fallbacks.get(action_name, f"Executing the {action_name} action.")
                
                structured_decision = {
                    "perception": "Instantaneous state evaluated via System One Decision Model.",
                    "thought": jev_thought,
                    "narrative": narrative_str,
                    "action": action_name,
                    "decision_model": "System One (Jev)",
                    "execution_layer": "Action Execution Layer (Deterministic Software)"
                }
                context.event_emitter("thought", structured_decision)

            if action_name not in allowed_actions:
                logging.warning(f"Invalid action '{action_name}'. Falling back.")
                action_name = next(iter(allowed_actions))

            env_action = context.adapter.translate_action(action_name)
            
            try:
                if action_name == "STOP":
                    obs_tuple = (rgb_array_observation, 0.0, True, False, {})
                else:
                    obs_tuple = context.env.step(env_action)
                    
                initial_obs, reward, terminated, truncated, info = obs_tuple
                total_reward += float(reward)
            except Exception as e:
                logging.error(f"Error executing environment step: {e}")
                break

            if config.RENDER_MODE in ("human", "record"):
                rgb_array_observation = context.env.unwrapped.render()
            else:
                rgb_array_observation = context.env.render()
                
            if context.event_emitter:
                context.event_emitter("frame", rgb_array_observation)

            logging.info(f"Executed action: {action_name} | Game Reward (Hidden): {reward}")
            step_count += 1
            
        # Determine true objective success: check player health and adapter-specific rules
        player_alive = True
        game = getattr(context.env.unwrapped, "game", None)
        if game is not None and hasattr(game, "get_game_variable"):
            try:
                import vizdoom
                player_alive = game.get_game_variable(vizdoom.GameVariable.HEALTH) > 0
            except Exception:
                pass

        if hasattr(context.adapter, "check_success"):
            success = context.adapter.check_success(total_reward, player_alive)
        else:
            success = total_reward > 0.0 and player_alive
        
        if terminated or truncated:
            status_narrative = context.adapter.get_termination_message(success)
            final_decision = {
                "perception": "The environment has signaled that the episode is over.",
                "thought": "The task has concluded.",
                "narrative": status_narrative,
                "action": "DONE"
            }
            if context.event_emitter:
                context.event_emitter("thought", final_decision)

        return {
            "success": success,
            "total_reward": total_reward,
            "steps": step_count,
        }
