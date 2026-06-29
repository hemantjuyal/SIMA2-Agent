import logging
import re
from typing import Any, Dict

from gsima.utils import config
from .base import BaseAgent
from .context import AgentContext
from gsima.agents.rollout import RolloutPlanner


def _parse_markdown_kv(markdown_text: str) -> Dict[str, str]:
    """
    Parses a simple markdown key-value list into a dictionary.
    This version is more robust to variations in model output.
    It handles optional bolding of keys and optional brackets around values.
    """
    data = {}
    for line in markdown_text.split('\n'):
        cleaned = line.strip()
        if not cleaned or ':' not in cleaned:
            continue

        cleaned = re.sub(r'^[*+-]\s*', '', cleaned)
        key, value = cleaned.split(':', 1)
        key = key.strip().strip('*').strip()
        value = value.strip().strip('[]').strip()
        if not key or not value:
            continue

        pythonic_key = re.sub(r'\s+', '_', key.lower())
        data[pythonic_key] = value

    return data


def get_decision_from_prompt(prompt: str, llm_runtime: Any) -> Dict[str, str]:
    """
    Returns a decision dictionary by invoking the provided text-only LLM runtime
    with a dynamic prompt and parsing the markdown response.
    """
    raw_response = llm_runtime.get_model_response(prompt)
    logging.info(f"Raw model response: {raw_response}")

    decision_dict = _parse_markdown_kv(raw_response)
    if not decision_dict:
        raise RuntimeError(f"LLM did not return any identifiable structured data in markdown format. Raw response: '{raw_response}'")
    
    if "action" not in decision_dict:
        raise RuntimeError(f"LLM response missing 'action' key in markdown structure. Raw response: '{raw_response}'")

    return decision_dict

class WorldModelAgent(BaseAgent):
    """
    A world model agent that implements the Perceive-Imagine-Plan-Act cycle.
    This agent uses distinct models for perception, simulating future states,
    and controlling actions based on imagined outcomes.
    """

    def run(self, context: AgentContext):
        """The main execution loop for the agent."""
        initial_obs, info = context.env.reset()
        context.memory_system.clear()
        logging.info("Environment reset and memory cleared.")

        # Initial observation
        initial_obs, info = context.env.reset()
        context.memory_system.clear()
        logging.info("Environment reset and memory cleared.")

        # Obtain the RGB array for VLM perception
        # We need to access the unwrapped env's render method for rgb_array if human/record mode is active,
        # because the HumanRendering/RecordVideo wrappers just display/save, but the underlying env still produces rgb_array.
        # If RENDER_MODE is already rgb_array, then context.env.render() directly returns it.
        if config.RENDER_MODE == "human" or config.RENDER_MODE == "record":
            rgb_array_observation = context.env.unwrapped.render()
        else: # This case is when config.RENDER_MODE is directly "rgb_array" for the top-level env
            rgb_array_observation = context.env.render()

        terminated, truncated, step_count = False, False, 0

        while not terminated and not truncated and step_count < config.MAX_STEPS:
            if config.RENDER_MODE == "human":
                context.env.render() # This is for displaying the human view
                
            logging.info(f"--- Step {step_count + 1}/{config.MAX_STEPS} ---")

            # 1. PERCEIVE the environment (using Perception VLM)
            structured_perception = {}
            vlm_raw_response = ""
            visual_prompt = context.get_visual_prompt()
            logging.info(f"Visual prompt:\n{visual_prompt}")
            try:
                # Use perception_runtime for VLM, passing the collected rgb_array_observation
                vlm_raw_response = context.perception_runtime.get_model_response(visual_prompt, rgb_array_observation)
                logging.info(f"VLM raw response: '{vlm_raw_response}'")
                structured_perception = _parse_markdown_kv(vlm_raw_response)
                structured_perception = context.adapter.normalize_perception(
                    structured_perception,
                    vlm_raw_response,
                )
                logging.info(f"VLM perception (parsed): {structured_perception}")
            except RuntimeError as e:
                logging.warning(f"Perception VLM failed: {e}. Proceeding without visual data for this step.")
                # structured_perception will remain empty

            # 2. IMAGINE & PLAN (using Simulator LLM and Controller LLM)
            memory_summary = context.memory_system.retrieve()

            # --- PERCEPTION Filtering: only keep semantic labels from VLM ---
            # Remove any keys that look like precise state (agent/goal/pos/orientation)
            if structured_perception:
                semantic_perception = {
                    k: v for k, v in structured_perception.items()
                    if not any(substr in k for substr in ['agent', 'goal', 'pos', 'orientation'])
                }
            else:
                semantic_perception = {}

            best_action_name = None
            ranked = []
            allowed_actions = {action.name for action in context.adapter.get_canonical_actions()}

            # Let the adapter inject any environment-specific action overrides.
            suggested_action = context.adapter.suggest_action(semantic_perception)
            if suggested_action is not None:
                if suggested_action in allowed_actions:
                    logging.info(
                        "Adapter suggested action '%s' based on environment-specific perception heuristics.",
                        suggested_action,
                    )
                    best_action_name = suggested_action
                else:
                    logging.warning(
                        "Adapter suggested unsupported action '%s'; ignoring suggestion.",
                        suggested_action,
                    )

            if not context.adapter.supports_simulation() and best_action_name is None:
                logging.info(
                    "Adapter does not support deterministic simulation; asking the controller to choose a direct action."
                )
                try:
                    controller_prompt = context.get_controller_prompt(
                        config.INSTRUCTION,
                        semantic_perception,
                        memory_summary,
                        {},
                    )
                    controller_decision = get_decision_from_prompt(
                        controller_prompt,
                        context.controller_runtime,
                    )
                    candidate_action = controller_decision.get('action', '').strip()
                    if candidate_action in allowed_actions:
                        best_action_name = candidate_action
                        logging.info(
                            "Controller selected action '%s' because simulation is unavailable.",
                            best_action_name,
                        )
                except Exception as e:
                    logging.warning(
                        "Controller direct action selection failed: %s. Falling back to adapter suggestion.",
                        e,
                    )
                if best_action_name is None:
                    fallback_action = context.adapter.suggest_action(semantic_perception)
                    if fallback_action in allowed_actions:
                        best_action_name = fallback_action
                        logging.info(
                            "Adapter fallback selected action '%s' because simulation is unavailable.",
                            best_action_name,
                        )

            if best_action_name is None and context.adapter.supports_simulation():
                # --- Imagine & Plan using deterministic simulator + rollout planner ---
                logging.info(
                    "Performing multi-step rollouts with deterministic simulator..."
                )
                # Create a planner configured from global config
                planner = RolloutPlanner(
                    adapter=context.adapter,
                    depth=config.ROLLOUT_DEPTH,
                    turn_penalty=config.TURN_PENALTY,
                    collision_penalty=config.COLLISION_PENALTY,
                    goal_bonus=config.GOAL_BONUS,
                )

                try:
                    best_action_name, ranked = planner.plan()
                except Exception as e:
                    logging.error(
                        f"Rollout planner failed: {e}. Falling back to direct controller selection."
                    )
                    try:
                        allowed_actions = {action.name for action in context.adapter.get_canonical_actions()}
                        controller_prompt = context.get_controller_prompt(
                            config.INSTRUCTION,
                            semantic_perception,
                            memory_summary,
                            {},
                        )
                        controller_decision = get_decision_from_prompt(
                            controller_prompt,
                            context.controller_runtime,
                        )
                        candidate_action = controller_decision.get('action', '').strip()
                        if candidate_action in allowed_actions:
                            best_action_name = candidate_action
                            logging.info(
                                "Controller selected action '%s' after planner failure.",
                                best_action_name,
                            )
                    except Exception as e2:
                        logging.warning(
                            "Controller fallback after planner failure also failed: %s", e2
                        )

            if best_action_name is None:
                fallback_action = context.adapter.get_fallback_action(semantic_perception)
                if fallback_action in allowed_actions:
                    best_action_name = fallback_action
                    logging.info(
                        "No planner/controller action available; using adapter fallback action '%s'.",
                        best_action_name,
                    )
                else:
                    best_action_name = next(iter(allowed_actions))
                    logging.info(
                        "No planner/controller action available; using default supported action '%s'.",
                        best_action_name,
                    )

            # Enforce STOP exposure rules using adapter capability metadata.
            progress = context.adapter.get_progress()
            dist_to_goal = progress.get('distance_to_goal') if isinstance(progress, dict) else None
            if dist_to_goal is None and context.adapter.supports_goal_distance():
                try:
                    current_state = context.adapter.get_current_env_state()
                    current_pos = current_state.get('agent_pos')
                    if current_pos is not None:
                        dist_to_goal = context.adapter.get_distance_to_goal(current_pos)
                except Exception:
                    dist_to_goal = None

            if (
                best_action_name == 'STOP'
                and context.adapter.supports_stop()
                and dist_to_goal is not None
                and dist_to_goal > config.STOP_DISTANCE_THRESHOLD
            ):
                logging.info("Best action was STOP but goal not in threshold; ignoring STOP and choosing alternative.")
                # pick next best non-STOP action
                for seq, traj, score in ranked:
                    first = seq[0] if seq else None
                    if first and first != 'STOP':
                        best_action_name = first
                        break

            # Ask controller LLM only for an explanation/thought about chosen action (not for action selection)
            thought = "N/A"
            if context.adapter.should_explain_action(best_action_name, semantic_perception):
                try:
                    explanation_prompt = context.get_controller_prompt(config.INSTRUCTION, semantic_perception, memory_summary, {})
                    # Append a short instruction to explain the chosen action
                    explanation_prompt += f"\n\nGiven that the planner selected: {best_action_name}, provide a one-sentence rationale for this choice. Return only a single markdown list item '- **thought**: [your one-sentence thought]'."
                    raw_explanation = context.controller_runtime.get_model_response(explanation_prompt)
                    parsed = _parse_markdown_kv(raw_explanation)
                    thought = parsed.get('thought', raw_explanation.strip())
                except Exception as e:
                    logging.warning(f"Controller explanation failed: {e}. Continuing without LLM thought.")
            else:
                thought = "Adapter-selected reflex action."

            action_name = best_action_name
            logging.info(f"Planner selected action: {action_name} (thought: {thought})")

            env_action = context.adapter.translate_action(action_name)
            if env_action == -1:  # STOP sentinel
                logging.info("STOP action received. Ending episode.")
                break

            # 3. INTERACT with the environment
            try:
                obs_tuple = context.env.step(env_action)
                observation, reward, terminated, truncated, info = obs_tuple
                
                # Validate that the environment step was executed
                logging.debug(f"Environment step executed. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            except Exception as e:
                logging.error(f"Error executing environment step with action '{action_name}': {e}")
                break
            
            outcome = context.get_outcome_from_reward(reward)

            # 4. LEARN from the experience as a structured transition
            structured_entry = {
                "step": step_count + 1,
                "action_name": action_name,
                "reward": float(reward),
            }
            for state_getter in (context.adapter.get_state_summary, context.adapter.get_progress):
                try:
                    state_data = state_getter()
                except Exception:
                    state_data = {}
                if isinstance(state_data, dict):
                    structured_entry.update(state_data)

            # Keep raw perception for debugging but do not use it as authoritative state
            structured_entry["perception_raw"] = vlm_raw_response
            structured_entry["thought"] = thought
            context.memory_system.add(structured_entry)

            initial_obs = observation
            
            # VLM perception is always used by the world model agent
            # Get fresh observation AFTER the step has been executed
            if not (terminated or truncated):
                if config.RENDER_MODE == "human" or config.RENDER_MODE == "record":
                    rgb_array_observation = context.env.unwrapped.render()
                else:
                    rgb_array_observation = context.env.render()
                logging.debug(f"Updated observation after step {step_count + 1}")

            logging.info(f"Executed action: {action_name}, Reward: {reward:.2f}, Outcome: {outcome}")
            step_count += 1
