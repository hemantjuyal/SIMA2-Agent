"""Builds the LLM prompt for MiniGrid environments."""
import logging
from typing import Deque, Dict, Any
from gsima.environments.minigrid.schema import SUPPORTED_ACTIONS

def get_visual_prompt() -> str:
    """
    Returns the environment-specific prompt string for the VLM to describe the scene.
    """
    return (
        "You are an agent in a simple grid-based world. "
        "The image provided is a top-down view of this world. "
        "You are the red triangle, and the direction you face is indicated by the triangle's point. "
        "Describe the scene concisely. Crucially, you must state what is directly in front of you (e.g., 'a wall', 'an open path', 'the green square'). "
        "Also, describe your general location and the location of the green square goal. "
        "Black squares are obstacles. Your description should be a few sentences at most."
    )

def get_prompt(instruction: str, visual_description: str = "", memory_summary: str = "") -> str:
    """
    Dynamically generates the prompt for the agent based on the supported
    actions for the MiniGrid environment, a visual description, and a memory summary.
    """
    action_descriptions = {
        "MOVE_FORWARD": "Move one step in the current direction.",
        "TURN_LEFT": "Rotate 90 degrees counter-clockwise to change direction.",
        "TURN_RIGHT": "Rotate 90 degrees clockwise to change direction.",
        "STOP": "Terminate the episode ONLY when you are on the green square."
    }
    action_list_with_descriptions = "\n".join([f"- {action.name}: {action_descriptions[action.name]}" for action in SUPPORTED_ACTIONS])
    
    prompt = f"""
You are an intelligent agent in a grid world. Your mission is to find the green square and then use the STOP action. Always aim for the most efficient path.

First, analyze your current situation based on the visual input.
Second, review your recent history to understand what has and has not worked.
Finally, decide your next action.

---
**Step 1: Analyze the Visual Situation**

Here is your current visual perception from your VLM:
"{visual_description}"

Based *only* on this visual description, answer the following:
- What is directly in front of you? (e.g., 'a wall', 'an open path', 'the green square')
- Where is the goal (green square) relative to you? (e.g., 'to my right', 'to my bottom-left')

---
**Step 2: Review Your Recent History**

This is what you have done in the last few steps:
{memory_summary}

---
**Step 3: Decide Your Next Action**

Based on your analysis in Step 1 and your history from Step 2, choose the single best action.

- If the path ahead is clear and moves you towards the goal, choose `MOVE_FORWARD`.
- If your path is blocked by an obstacle, you *must* turn. Choose `TURN_LEFT` or `TURN_RIGHT`.
- If you are on the green square, you *must* choose `STOP`.

Available Actions:
{action_list_with_descriptions}

Return your answer as a single, valid JSON object with only the "action" key.
Example: {{"action": "MOVE_FORWARD"}}

Now, provide ONLY the JSON object for your chosen action.
JSON:
"""
    return prompt

def get_outcome_from_reward(reward: float) -> str:
    """Translates a numeric reward into a human-readable outcome for MiniGrid."""
    if reward > 0.0:
        return "Good move (closer to goal)"
    elif reward < -0.05:
        return "Bad move (hit an obstacle)"
    else:
        return "Inefficient move (no progress)"

def create_memory_summary(memory: Deque[Dict[str, Any]]) -> str:
    """Creates a summarized string of the agent's recent memory for MiniGrid."""
    if not memory:
        return "No history yet."
    
    summary = (
        "\nYour goal is to reach the green square. Here is a summary of your last few actions:\n"
    )
    
    for entry in memory:
        summary += (
            f"  - Step {entry['step']}: You chose '{entry['action_name']}'. "
            f"Outcome: {entry['outcome']} (Reward: {entry['reward']:.2f}).\n"
        )
        
    summary += "\nConsidering this history, what is the optimal action to take next?\n"
    return summary
