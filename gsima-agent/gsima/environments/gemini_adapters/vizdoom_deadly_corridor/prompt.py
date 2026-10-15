"""Prompts and summaries for ViZDoom Deadly Corridor environment."""
from typing import Any, Deque, Dict
from gsima.environments.gemini_adapters.vizdoom_deadly_corridor.schema import SUPPORTED_ACTIONS

def get_multimodal_prompt(instruction: str, memory_summary: str) -> str:
    """Create a multimodal prompt for the ViZDoom Deadly Corridor action selection loop."""
    action_list = [action.name for action in SUPPORTED_ACTIONS]

    return f"""You are an agent controlling a first-person shooter environment.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** the image provided (your current visual perception).
2.  **Read** your MEMORY to avoid repeating mistakes or loops.
3.  **Think** about the best tactical move to eliminate threatening monsters or advance toward the green vest.
4.  **Act** by choosing a single action.

---
**MEMORY (Recent History):**
{memory_summary}

---
**DECISION RULES:**
- You are in a narrow corridor with monsters on both sides and a green armor vest at the far end.
- Use TURN_LEFT or TURN_RIGHT to aim your weapon horizontally.
- Use SHOOT to eliminate threatening monsters in your path.
- Use MOVE_FORWARD to advance down the corridor toward the green vest.
- Balance combat and forward progress to avoid taking excessive damage.

Return your decision in a strict markdown list format with these exactly 4 keys:
- **perception**: [Briefly describe what you see: monsters, corridors, green vest, crosshair alignment]
- **thought**: [State your rationale: whether to shoot threatening enemies or advance toward the vest]
- **narrative**: [A confident first-person sentence declaring your action]
- **action**: [The exact action to execute. Must be exactly one of: {action_list}]
"""

def create_memory_summary(memory: Deque[Dict[str, Any]]) -> str:
    """Summarize recent transition history."""
    if not memory:
        return "No history yet."

    summary = "Recent thoughts and actions:\n"
    for entry in memory:
        step = entry.get("step", "?")
        thought = entry.get("thought", "N/A")
        action = entry.get("action_name", entry.get("action", "N/A"))
        summary += f"- Step {step}: Thought=[{thought}] -> Action=[{action}]\n"

    return summary
