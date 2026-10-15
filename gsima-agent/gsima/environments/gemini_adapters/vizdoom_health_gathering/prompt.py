"""Prompts and summaries for ViZDoom Health Gathering environment (Gemini Runtime)."""
from typing import Any, Deque, Dict
from gsima.environments.gemini_adapters.vizdoom_health_gathering.schema import SUPPORTED_ACTIONS

def get_multimodal_prompt(instruction: str, memory_summary: str) -> str:
    """Create a multimodal prompt for the ViZDoom Health Gathering action selection loop."""
    action_list = [action.name for action in SUPPORTED_ACTIONS]

    return f"""You are an agent controlling a first-person shooter environment.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** the image provided (your current visual perception).
2.  **Read** your MEMORY to avoid repeating mistakes or loops.
3.  **Think** about the best move to navigate toward visible medikits to replenish health and survive the acidic floor.
4.  **Act** by choosing a single action.

---
**MEMORY (Recent History):**
{memory_summary}

---
**DECISION RULES:**
- You are in a large chamber with an acidic floor that slowly drains your health.
- White medical kits with red crosses are scattered around the chamber. Collecting one restores health (+25 HP).
- Use TURN_LEFT or TURN_RIGHT to center your view on a visible medikit.
- Use MOVE_FORWARD to approach and pick up the medikit.
- If no medikits are in view, use TURN_LEFT or TURN_RIGHT to scan around the chamber.

Return your decision in a strict markdown list format with these exactly 4 keys:
- **perception**: [Briefly describe what you see: visible medikits, distance, orientation]
- **thought**: [State your rationale: whether turning to center a medikit or moving forward to collect it]
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
