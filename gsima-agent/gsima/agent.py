import json
import logging
import re
from typing import Any, Dict, Optional, Deque
from collections import deque
import numpy as np # Import numpy for type hinting
import gymnasium as gym

from gsima import runtime
from gsima.utils import config # Import config to check AGENT_MODEL_TYPE
from gsima.runtime.base import BaseModelRuntime
from gsima.environments.base import BaseAdapter

# Helper function for safe JSON extraction
def _extract_json_safely(raw_response: str) -> Optional[str]:
    """
    Attempts to extract a JSON string from a raw LLM response.
    Returns the extracted JSON string or None if not found/invalid.
    """
    # Initial cleaning: strip whitespace and replace single quotes (common LLM issue)
    cleaned_response = raw_response.strip().replace("'", '"')

    # Try to find a JSON object embedded in the cleaned response using regex
    # This regex looks for an opening curly brace, any characters, and a closing curly brace
    # It's made non-greedy to catch the smallest valid JSON object
    json_match = re.search(r"\{.*?\}", cleaned_response, re.DOTALL)
    
    if json_match:
        logging.debug(f"Extracted JSON with regex: {json_match.group(0)}")
        return json_match.group(0)
    
    # Fallback: check for ```json``` block (less robust but covers some cases)
    if "```json" in cleaned_response:
        try:
            json_str_from_block = cleaned_response.split("```json")[1].split("```")[0]
            logging.debug(f"Extracted JSON from ```json``` block: {json_str_from_block}")
            return json_str_from_block
        except IndexError:
            logging.debug("Failed to extract JSON from ```json``` block due to IndexError.")
            
    # If no specific JSON found, assume the whole cleaned response is intended JSON
    logging.debug(f"No specific JSON found, attempting to parse full cleaned response: {cleaned_response}")
    return cleaned_response if cleaned_response else None

def get_action_from_prompt(prompt: str, llm_runtime: Any) -> Dict:
    """
    Returns a JSON action by invoking the provided text-only LLM runtime
    with a dynamic prompt.
    """
    raw_response = llm_runtime.get_model_response(prompt)
    logging.debug(f"Raw model response: {raw_response}")

    json_str_to_parse: Optional[str] = None
    action_json: Dict = {}

    try:
        json_str_to_parse = _extract_json_safely(raw_response)

        if json_str_to_parse is None:
            logging.error(f"LLM did not return any identifiable JSON structure. Raw response: '{raw_response}'")
            raise RuntimeError("LLM did not return valid JSON")
        
        action_json = json.loads(json_str_to_parse)
        
        if "action" not in action_json:
            logging.error(f"Parsed JSON missing 'action' key: {action_json}. Raw response: '{raw_response}'")
            raise RuntimeError("LLM response missing 'action' key")

        return action_json

    except json.JSONDecodeError:
        logging.error(f"Failed to decode extracted JSON: '{json_str_to_parse}'. Raw response: '{raw_response}'")
        raise RuntimeError(f"LLM returned malformed JSON: '{json_str_to_parse}'")
    except RuntimeError:  # Re-raise if our internal RuntimeError
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during action parsing: {e}. Raw response: '{raw_response}'")
        raise RuntimeError(f"Unexpected error during action parsing: {e}")

def run_agent_loop(
    env: gym.Env, 
    adapter: BaseAdapter, 
    vlm_runtime: BaseModelRuntime, 
    llm_runtime: BaseModelRuntime,
    get_prompt: callable,
    get_visual_prompt: callable,
    create_memory_summary: callable,
    get_outcome_from_reward: callable,
):
    """The main execution loop for the agent."""
    initial_obs, info = env.reset()
    logging.info("Environment reset.")

    # Get the initial image observation to start the loop
    image_observation = env.render() if config.AGENT_MODEL_TYPE == "vlm" else None
    
    memory: Deque[Dict[str, Any]] = deque(maxlen=config.MEMORY_LENGTH)
    
    terminated, truncated, step_count = False, False, 0

    while not terminated and not truncated and step_count < config.MAX_STEPS:
        if config.RENDER_MODE == "human":
            env.render()
            
        logging.info(f"--- Step {step_count + 1}/{config.MAX_STEPS} ---")

        # 1. Get a description of the current state's image
        visual_description = ""
        if config.AGENT_MODEL_TYPE == "vlm":
            visual_prompt = get_visual_prompt()
            visual_description = vlm_runtime.get_model_response(visual_prompt, image_observation)
            logging.info(f"Visual description: {visual_description}")

        
        # 2. Decide on an action based on the current state and memory
        memory_summary = create_memory_summary(memory)
        prompt = get_prompt(config.INSTRUCTION, visual_description, memory_summary)
        
        action_json = get_action_from_prompt(prompt, llm_runtime)
        logging.info(f"Agent decided: {action_json}")

        try:
            action_name = action_json["action"]
            env_action = adapter.translate_action(action_name)
            if env_action == -1: # STOP sentinel
                logging.info("STOP action received. Ending episode.")
                break
        except (KeyError, TypeError) as e:
            logging.error(f"Invalid action format: {action_json}. Ending episode. Error: {e}")
            break

        # 3. Execute the action and get the new state and reward
        try:
            obs_tuple = env.step(env_action)
            observation, reward, terminated, truncated, info = obs_tuple
        except Exception as e:
            logging.error(f"Error executing environment step with action '{action_name}': {e}")
            terminated = True
            observation, reward, truncated, info = initial_obs, 0, True, {}
        
        outcome = get_outcome_from_reward(reward)

        # 4. Store the memory of the action taken in the *previous* state
        memory.append({
            "step": step_count + 1, "action_name": action_name,
            "reward": reward, "visual_description": visual_description,
            "outcome": outcome,
        })

        initial_obs = observation
        
        # 5. Get the image of the *new* state for the next loop iteration
        if config.AGENT_MODEL_TYPE == "vlm" and not (terminated or truncated):
            image_observation = env.render()

        logging.info(f"Executed action: {action_name}, Reward: {reward:.2f}, Outcome: {outcome}")
        step_count += 1


