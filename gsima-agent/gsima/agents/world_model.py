import logging
import re
from typing import Any, Dict

import gymnasium as gym

from gsima.utils import config
from .base import BaseAgent
from .context import AgentContext


def _parse_markdown_kv(markdown_text: str) -> Dict[str, str]:
    """
    Parses a simple markdown key-value list into a dictionary.
    This version is more robust to variations in model output.
    It handles optional bolding of keys and optional brackets around values.
    """
    data = {}
    # Regex to find key-value pairs, more forgiving.
    pattern = re.compile(r"-\s*\**(.+?)\**\s*:\s*\[?(.+?)\]?\s*?$")
    
    for line in markdown_text.split('\n'):
        match = pattern.match(line.strip())
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            
            # Create a more pythonic key
            pythonic_key = key.lower().replace(' ', '_')
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
                logging.info(f"VLM perception (parsed): {structured_perception}")
            except RuntimeError as e:
                logging.warning(f"Perception VLM failed: {e}. Proceeding without visual data for this step.")
                # structured_perception will remain empty

            # 2. IMAGINE & PLAN (using Simulator LLM and Controller LLM)
            memory_summary = context.memory_system.retrieve()
            
            # --- Imagine Phase (using Deterministic Simulator) ---
            logging.info("Imagining future states with deterministic simulator...")
            imagined_futures = {}
            possible_actions = context.adapter.get_canonical_actions()
            
            for action in possible_actions:
                try:
                    simulated_next_state = context.adapter.simulate_next_state(action.name)
                    imagined_futures[action.name] = simulated_next_state
                    logging.debug(f"Simulated next state for '{action.name}': {simulated_next_state}")
                except Exception as e:
                    logging.error(f"Deterministic simulator failed for action '{action.name}': {e}")
                    # If simulator fails, that action's future is not imagined
            
            # --- Plan Phase ---
            controller_prompt = context.get_controller_prompt(config.INSTRUCTION, structured_perception, memory_summary, imagined_futures)
            logging.info(f"Controller LLM prompt:\n{controller_prompt}")
            
            try:
                # Use controller_runtime for action decision
                decision_dict = get_decision_from_prompt(controller_prompt, context.controller_runtime)
                thought = decision_dict.get("thought", "N/A")
                action_name = decision_dict.get("action")

                logging.info(f"Agent thought: \"{thought}\"")
                logging.info(f"Agent decided action: {action_name}")

                if not action_name:
                    raise KeyError("LLM response is missing the 'action' key.")

                env_action = context.adapter.translate_action(action_name)
                if env_action == -1: # STOP sentinel
                    logging.info("STOP action received. Ending episode.")
                    break
            except (RuntimeError, KeyError) as e:
                logging.error(f"Error during agent decision phase: {e}. Ending episode.")
                break

            # 3. INTERACT with the environment
            try:
                obs_tuple = context.env.step(env_action)
                observation, reward, terminated, truncated, info = obs_tuple
            except Exception as e:
                logging.error(f"Error executing environment step with action '{action_name}': {e}")
                break
            
            outcome = context.get_outcome_from_reward(reward)

            # 4. LEARN from the experience
            experience = {
                "step": step_count + 1,
                "thought": thought,
                "action_name": action_name,
                "reward": reward,
                "outcome": outcome,
                "perception_raw": vlm_raw_response, # Store the raw perception for debugging
            }
            context.memory_system.add(experience)

            initial_obs = observation
            
            # VLM perception is always used by the world model agent
            if not (terminated or truncated):
                if config.RENDER_MODE == "human" or config.RENDER_MODE == "record":
                    rgb_array_observation = context.env.unwrapped.render()
                else:
                    rgb_array_observation = context.env.render()

            logging.info(f"Executed action: {action_name}, Reward: {reward:.2f}, Outcome: {outcome}")
            step_count += 1
