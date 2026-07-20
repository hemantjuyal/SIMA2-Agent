import logging
import re
from typing import Any, Dict

from gsima.utils import config
from .base import BaseAgent
from .context import AgentContext


def _parse_markdown_kv(markdown_text: str) -> Dict[str, str]:
    """
    Parses a simple markdown key-value list into a dictionary.
    """
    data = {}
    for line in markdown_text.split('\n'):
        cleaned = line.strip()
        if not cleaned or ':' not in cleaned:
            continue

        cleaned = re.sub(r'^[*+-]\s*', '', cleaned)
        key, value = cleaned.split(':', 1)
        key = key.strip().strip('*').strip()
        value = value.strip().strip('[]\'"`*').strip()
        if not key or not value:
            continue

        pythonic_key = re.sub(r'\s+', '_', key.lower())
        data[pythonic_key] = value

    return data


class WorldModelAgent(BaseAgent):
    """
    A unified multimodal agent that implements the true SIMA 2 Perceive-Think-Act cycle.
    """

    def run(self, context: AgentContext):
        """The main execution loop for the agent."""
        if hasattr(config, "ENV_SEED") and config.ENV_SEED is not None:
            initial_obs, info = context.env.reset(seed=config.ENV_SEED)
        else:
            initial_obs, info = context.env.reset()
            
        context.memory_system.clear()
        logging.info("Environment reset and memory cleared.")

        # Obtain the RGB array for Gemini perception
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

            # 1. Prepare memory and prompt
            memory_summary = context.memory_system.retrieve()
            multimodal_prompt = context.get_multimodal_prompt(config.INSTRUCTION, memory_summary)
            logging.info(f"Unified Multimodal Prompt:\n{multimodal_prompt}")

            is_delay_phase = info.get("is_delay_phase", False)
            
            if is_delay_phase:
                import time
                time.sleep(0.05) # Add small delay to let UI play the death animation smoothly
                
                # Emit a kill confirmation on the first delay frame
                if not getattr(context, 'has_confirmed_kill', False):
                    context.has_confirmed_kill = True
                    structured_decision = {
                        "perception": "The monster has been eliminated.",
                        "thought": "The target is destroyed and the area is secure.",
                        "narrative": "Target eliminated. Mission accomplished.",
                        "action": "STOP"
                    }
                    if context.event_emitter:
                        context.event_emitter("thought", structured_decision)
                        
                structured_decision = {"thought": "Delay phase", "action": "NOOP"}
            else:
                try:
                    raw_response = context.multimodal_runtime.get_model_response(multimodal_prompt, rgb_array_observation)
                    logging.info(f"Gemini Raw Response:\n{raw_response}")
                    structured_decision = _parse_markdown_kv(raw_response)
                except Exception as e:
                    logging.error(f"Gemini Multimodal API failed: {e}")
                    # Fallback to random action if API fails
                    structured_decision = {"thought": "API Failed", "action": next(iter(allowed_actions))}

            # 3. Parse action and thought
            action_name = structured_decision.get("action", "").upper().strip().strip('\'"`*')
            thought = structured_decision.get("thought", "N/A").strip().strip('\'"`*')
            perception_text = structured_decision.get("perception", "N/A")
            
            # Robust fallback: if Gemini omits the narrative key, gracefully inject a natural language action.
            if "narrative" not in structured_decision or not structured_decision["narrative"].strip():
                action_fallbacks = {
                    "MOVE_FORWARD": "I am moving forward.",
                    "TURN_LEFT": "I am turning left.",
                    "TURN_RIGHT": "I am turning right.",
                    "PICK_UP": "I am picking up the object.",
                    "DROP": "I am dropping the object.",
                    "TOGGLE": "I am interacting with the object.",
                    "SHOOT": "I am firing my weapon.",
                }
                structured_decision["narrative"] = action_fallbacks.get(action_name, f"I am executing the {action_name} action.")
            
            if context.event_emitter and not is_delay_phase:
                context.event_emitter("thought", structured_decision)

            # Validate action
            if action_name not in allowed_actions:
                logging.warning(f"Gemini suggested invalid action '{action_name}'. Falling back.")
                action_name = next(iter(allowed_actions))

            env_action = context.adapter.translate_action(action_name)
            
            # 4. INTERACT with environment
            try:
                if action_name == "STOP":
                    obs_tuple = (rgb_array_observation, 0.0, True, False, {})
                else:
                    obs_tuple = context.env.step(env_action)
                    
                observation, reward, terminated, truncated, info = obs_tuple
                total_reward += float(reward)
            except Exception as e:
                logging.error(f"Error executing environment step: {e}")
                break

            # 5. LEARN from experience (No Game Engine Rewards allowed in memory!)
            if not is_delay_phase:
                structured_entry = {
                    "step": step_count + 1,
                    "perception": perception_text,
                    "thought": thought,
                    "action_name": action_name,
                }
                context.memory_system.add(structured_entry)
            
            # Always render the frame, even if terminated, to capture the final state
            if config.RENDER_MODE in ("human", "record"):
                rgb_array_observation = context.env.unwrapped.render()
            else:
                rgb_array_observation = context.env.render()
                
            if context.event_emitter:
                context.event_emitter("frame", rgb_array_observation)

            logging.info(f"Executed action: {action_name} | Game Reward (Hidden): {reward}")
            step_count += 1
            
        success = total_reward > 0.0
        
        # Broadcast a final status message to the UI
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
