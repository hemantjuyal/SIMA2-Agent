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
    Example: "- **Key**: [Value]" -> {"Key": "Value"}
    """
    data = {}
    # Regex to find key-value pairs within brackets
    pattern = re.compile(r"-\s*\*\*(.*?)\*\*:\s*\[(.*?)\]")
    matches = pattern.findall(markdown_text)
    
    for key, value in matches:
        clean_key = key.strip()
        clean_value = value.strip()
        # Create a more pythonic key, e.g., "Agent Orientation" -> "agent_orientation"
        pythonic_key = clean_key.lower().replace(' ', '_')
        data[pythonic_key] = clean_value
        
    return data

def get_decision_from_prompt(prompt: str, llm_runtime: Any) -> Dict[str, str]:
    """
    Returns a decision dictionary by invoking the provided text-only LLM runtime
    with a dynamic prompt and parsing the markdown response.
    """
    raw_response = llm_runtime.get_model_response(prompt)
    logging.debug(f"Raw model response: {raw_response}")

    decision_dict = _parse_markdown_kv(raw_response)
    if not decision_dict:
        raise RuntimeError("LLM did not return any identifiable structured data in markdown format")
    
    if "action" not in decision_dict:
        raise RuntimeError("LLM response missing 'action' key in markdown structure")

    return decision_dict

class SimpleLoopAgent(BaseAgent):
    """
    A reactive agent that operates in a step-by-step observation-action loop.
    This contains the primary execution logic.
    """

    def run(self, context: AgentContext):
        """The main execution loop for the agent."""
        initial_obs, info = context.env.reset()
        context.memory_system.clear()
        logging.info("Environment reset and memory cleared.")

        image_observation = context.env.render() if config.AGENT_MODEL_TYPE == "vlm" else None
        
        terminated, truncated, step_count = False, False, 0

        while not terminated and not truncated and step_count < config.MAX_STEPS:
            if config.RENDER_MODE == "human":
                context.env.render()
                
            logging.info(f"--- Step {step_count + 1}/{config.MAX_STEPS} ---")

            # 1. PERCEIVE the environment
            structured_perception = {}
            vlm_raw_response = ""
            if context.vlm_runtime:
                visual_prompt = context.get_visual_prompt()
                vlm_raw_response = context.vlm_runtime.get_model_response(visual_prompt, image_observation)
                structured_perception = _parse_markdown_kv(vlm_raw_response)
                logging.info(f"VLM perception (parsed): {structured_perception}")

            # 2. THINK and ACT
            memory_summary = context.memory_system.retrieve()
            # Pass the parsed dictionary to the prompt template
            prompt = context.get_prompt(config.INSTRUCTION, structured_perception, memory_summary)
            
            try:
                decision_dict = get_decision_from_prompt(prompt, context.llm_runtime)
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
            
            if context.vlm_runtime and not (terminated or truncated):
                image_observation = context.env.render()

            logging.info(f"Executed action: {action_name}, Reward: {reward:.2f}, Outcome: {outcome}")
            step_count += 1
