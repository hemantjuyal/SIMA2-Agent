"""
Prompts and summaries for MiniGrid environments, designed for a 'Perceive-Think-Act' agent.
"""
import json
from typing import Deque, Dict, Any

from gsima.environments.minigrid.schema import SUPPORTED_ACTIONS

def get_visual_prompt() -> str:
    """
    Returns a prompt that instructs the VLM to return a simple, human-readable
    Markdown list. This is more reliable for models that struggle with JSON.
    """
    return (
        "Analyze the provided grid world image from a top-down perspective. "
        "You are the red triangle. The goal is the green square. "
        "Describe the scene using the following markdown format. Do not add any other explanations or text. "
        "Provide your analysis inside the brackets. "
        "\n\n"
        "### Scene Analysis\n"
        "- **Agent Orientation**: [Your analysis: NORTH, SOUTH, EAST, or WEST]\n"
        "- **Goal Relative Position**: [Your analysis: e.g., front-left, right, back]\n"
        "- **Obstacle In Front**: [Your analysis: true or false]"
    )

def get_prompt(instruction: str, structured_perception: Dict[str, Any], memory_summary: str) -> str:
    """
    Dynamically generates the main prompt for the LLM, instructing it to think
    before it acts.
    """
    action_list = [action.name for action in SUPPORTED_ACTIONS]
    perception_str = json.dumps(structured_perception, indent=2)

    return f"""You are an intelligent and methodical agent in a grid world. Your mission is to efficiently reach the green square.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** your current situation from PERCEPTION and MEMORY.
2.  **THINK** and formulate a clear, one-sentence rationale for your next move.
3.  **ACT** by choosing a single action based on your thought.

---
**1. PERCEPTION (from Vision Model):**
```json
{perception_str}
```

**2. MEMORY (Recent History):**
{memory_summary}

---
**3. THINKING:**
Based on your MISSION, PERCEPTION, and MEMORY, formulate a short-term plan or rationale for your next move. Your thought must be a single, concise sentence.

**4. ACTION:**
Based on your THINKING, choose the single best action to take right now.

Return your decision as a single, valid JSON object with two keys: "thought" and "action".
- The "thought" value MUST be your one-sentence rationale from step 3.
- The "action" value MUST be one of: {action_list}

Example:
{{
  "thought": "My perception shows the goal is to my front-left and my path is clear, so I will turn left to face it.",
  "action": "TURN_LEFT"
}}

Now, provide ONLY the JSON object for your decision.
JSON:
"""

def get_outcome_from_reward(reward: float) -> str:
    """Translates a numeric reward into a human-readable outcome for MiniGrid."""
    if reward > 0.0:
        return "Good move (closer to goal)"
    elif reward < -0.05:
        return "Bad move (hit an obstacle)"
    else:
        return "Inefficient move (no progress)"

def create_memory_summary(memory: Deque[Dict[str, Any]]) -> str:
    """
    Creates a summarized string of the agent's recent memory, including thoughts
    and quantitative rewards.
    """
    if not memory:
        return "No history yet."
    
    summary = "This is a summary of your last few steps:\n"
    
    for entry in memory:
        # Round the reward for cleaner display
        reward_str = f"{entry['reward']:.2f}"
        summary += (
            f"- Step {entry['step']}: "
            f"You thought: \"{entry['thought']}\" | "
            f"You chose: '{entry['action_name']}' | "
            f"Outcome: {entry['outcome']} (Reward: {reward_str})\n"
        )
        
    summary += "Use this history to inform your next thought."
    return summary
