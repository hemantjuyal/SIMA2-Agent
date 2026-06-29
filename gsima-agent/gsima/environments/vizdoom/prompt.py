"""Prompts and summaries for ViZDoom environments."""
from typing import Any, Deque, Dict

from gsima.environments.vizdoom.schema import SUPPORTED_ACTIONS


def get_visual_prompt() -> str:
    """Prompt the VLM to describe a first-person combat scene."""
    return (
        "You are a semantic image interpreter for a first-person shooter game. "
        "Describe only semantic facts that the simulator may not know. "
        "Do NOT report exact coordinates or precise pixel counts. "
        "Return a short markdown list with any of the following keys if present: "
        "Monster, Move Direction, Kill Action, Environment Description. "
        "Examples: '- **Monster**: a monster in the left side not right at front', '- **Move Direction**: moving left or right to come right at front to the monster for efficient kill?', '- **Kill Action**: move left or right or shoot for efficient kill?', '- **Environment Description**: a rectangular room with brown walls monster is at right side'."
    )


def get_controller_prompt(
    instruction: str,
    structured_perception: Dict[str, Any],
    memory_summary: str,
    imagined_futures: Dict[str, Dict[str, str]],
) -> str:
    """Create a controller prompt for the ViZDoom action selection loop."""
    action_list = [action.name for action in SUPPORTED_ACTIONS]

    perception_items = [
        f"- {key.replace('_', ' ').title()}: {value}"
        for key, value in structured_perception.items()
    ]
    perception_str = "\n".join(perception_items) if perception_items else "No visual data available."

    return f"""You are an agent controlling a first-person shooter environment.

**MISSION:** {instruction}

**PERCEPTION:**
{perception_str}

**MEMORY:**
{memory_summary}

Choose bests actions from: {action_list}.
Prefer relevant and best action for efficient killing.

Return only a markdown list with:
- **thought**: one short sentence mentioning move direction or shot action, choose best action for better outcome.
- **action**: allowed actions to take, from the list above, in a single line, separated by commas.

Use MOVE_LEFT or MOVE_RIGHT when the perception indicates the monster is not right at front or when the scene recommends repositioning first to move towards the monster. Use SHOOT only when monster target is right at front and a clear opportunity to fire.
"""


def get_outcome_from_reward(reward: float) -> str:
    """Translate a reward into a concise outcome string."""
    if reward > 0:
        return "Good move (positive reward)"
    if reward < 0:
        return "Bad move (penalty)"
    return "Neutral move"


def create_memory_summary(memory: Deque[Dict[str, Any]]) -> str:
    """Summarize the recent transition history for the controller."""
    if not memory:
        return "No history yet."

    summary = "Recent transitions:\n"
    for entry in memory:
        step = entry.get("step", "?")
        action = entry.get("action_name", "N/A")
        reward = entry.get("reward", None)
        summary += f"- Step {step}: action={action} reward={reward}\n"
    return summary
