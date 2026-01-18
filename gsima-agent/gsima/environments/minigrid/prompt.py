"""
Prompts and summaries for MiniGrid environments, designed for a 'Perceive-Think-Act' agent.
"""
from typing import Deque, Dict, Any

from gsima.environments.minigrid.schema import SUPPORTED_ACTIONS

def get_visual_prompt() -> str:
    """
    Returns a prompt that instructs the VLM to return a simple, human-readable
    Markdown list. This is more reliable for models that struggle with JSON.
    This version is simplified to be more of a Q&A format for smaller models.
    """
    return (
        "You are a helpful assistant analyzing an image from a grid-world game. "
        "You are the red triangle. The goal is the green square. "
        "Please answer the following questions about the scene in a markdown list. "
        "\n\n"
        "- **Agent Orientation**: [What direction is the red triangle pointing? (NORTH, SOUTH, EAST, or WEST)]\n"
        "- **Goal Relative Position**: [Where is the green square relative to the agent? (e.g., front, front-left, back-right)]\n"
        "- **Obstacle In Front**: [Is there a wall or obstacle directly in front of the agent? (true or false)]"
    )

def get_prompt(instruction: str, structured_perception: Dict[str, Any], memory_summary: str) -> str:
    """
    Dynamically generates the main prompt for the LLM, instructing it to think
    before it acts.
    """
    action_list = [action.name for action in SUPPORTED_ACTIONS]
    # Format the perception dictionary as a simple key-value string for the prompt
    perception_items = [f"- {key.replace('_', ' ').title()}: {value}" for key, value in structured_perception.items()]
    perception_str = "\n".join(perception_items) if perception_items else "No visual data available."

    return f"""You are an intelligent and methodical agent in a grid world. Your mission is to efficiently reach the green square.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** your current situation from PERCEPTION and MEMORY.
2.  **THINK** and formulate a clear, one-sentence rationale for your next move.
3.  **ACT** by choosing a single action based on your thought.

---
**1. PERCEPTION (from Vision Model):**
{perception_str}

**2. MEMORY (Recent History):**
{memory_summary}

---
**3. THINKING AND ACTING:**
Based on your MISSION, PERCEPTION, and MEMORY, provide your thought process and the single best action to take right now.

Return your decision in a markdown list format.
- The "thought" value MUST be your one-sentence rationale.
- The "action" value MUST be one of: {action_list}

Example:
- **thought**: [My perception shows the goal is to my front-left and my path is clear, so I will turn left to face it.]
- **action**: [TURN_LEFT]

Now, provide ONLY the markdown for your decision.
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
