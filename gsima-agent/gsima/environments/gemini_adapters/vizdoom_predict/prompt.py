"""Prompts and summaries for ViZDoom environments."""
from typing import Any, Deque, Dict
from gsima.environments.vizdoom_predict.schema import SUPPORTED_ACTIONS

def get_multimodal_prompt(instruction: str, memory_summary: str) -> str:
    """Create a multimodal prompt for the ViZDoom action selection loop."""
    action_list = [action.name for action in SUPPORTED_ACTIONS]

    return f"""You are an agent controlling a first-person shooter environment.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** the image provided (your current visual perception).
2.  **Read** your MEMORY to avoid repeating mistakes or loops.
3.  **Think** about the best move to align with the monster and kill it.
4.  **Act** by choosing a single action.

---
**MEMORY (Recent History):**
{memory_summary}

---
**DECISION RULES:**
- The target is moving left or right. You must predict its position.
- Use TURN_LEFT or TURN_RIGHT to aim your weapon horizontally.
- **AGGRESSIVE FIRING RULE:** Do NOT wait for perfect alignment! The missile travels slowly and has a blast radius. If the monster is *anywhere* near the center third of your screen, you MUST choose SHOOT to lead the target.
- It is much better to fire multiple missiles slightly off-center than to waste time turning perfectly.

Return your decision in a strict markdown list format with these exactly 4 keys:
- **perception**: [Briefly describe the moving monster's position and trajectory relative to your crosshairs]
- **thought**: [Based strictly on the perception, state your rationale for tracking the target or predicting its path to fire]
- **narrative**: [Based on the thought, write a confident, tactical first-person sentence declaring your action. Example: "I am predicting its path and firing a missile to intercept it!"]
- **action**: [The exact combat action that executes the thought. Must be exactly one of: {action_list}]
"""

def create_memory_summary(memory: Deque[Dict[str, Any]]) -> str:
    """Summarize the recent transition history containing only thoughts and actions."""
    if not memory:
        return "No history yet."

    summary = "Recent thoughts and actions:\n"
    for entry in memory:
        step = entry.get("step", "?")
        thought = entry.get('thought', 'N/A')
        action = entry.get("action_name", entry.get("action", "N/A"))
        summary += f"- Step {step}: Thought=[{thought}] -> Action=[{action}]\n"
        
    return summary
