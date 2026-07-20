"""
Prompts and summaries for MiniGrid environments, designed for a Multimodal SIMA2 agent.
"""
from typing import Deque, Dict, Any
from gsima.environments.minigrid_empty.schema import SUPPORTED_ACTIONS

def get_multimodal_prompt(instruction: str, memory_summary: str) -> str:
    """
    Dynamically generates the main prompt for the Multimodal Gemini model.
    """
    action_list = [action.name for action in SUPPORTED_ACTIONS]
    
    return f"""You are an intelligent and methodical agent in a grid world. Your mission is to efficiently reach the green square.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** the image provided (your current visual perception).
2.  **Read** your MEMORY to avoid repeating mistakes or loops.
3.  **Think** and formulate a clear, one-sentence rationale for your next move based on your MISSION and what you see.
4.  **Act** by choosing a single action.

---
**MEMORY (Recent History):**
{memory_summary}

---
**DECISION RULES:**
1. **Do NOT choose an action that moves you into a wall or obstacle.**
2. **Evaluate goal position**: Choose actions that orient you toward or move you closer to the green square.
3. **Do NOT repeat ineffective actions** - Learn from memory if you are stuck in a loop.
4. **Do NOT simply repeat thoughts from previous steps** - Every step is a new situation.

Return your decision in a strict markdown list format with these exactly 4 keys:
- **perception**: [Briefly describe your orientation and the location of the goal square]
- **thought**: [Based strictly on the perception, state your spatial reasoning for the next move]
- **narrative**: [Based on the thought, write an engaging, first-person sentence declaring your action. Example: "I am turning left to face the goal square."]
- **action**: [The exact action that executes the thought. Must be exactly one of: {action_list}]
"""

def create_memory_summary(memory: Deque[Dict[str, Any]]) -> str:
    """
    Creates a summarized string of the agent's recent memory, including ONLY thoughts and actions.
    No internal game rewards or distances!
    """
    if not memory:
        return "No history yet."

    summary = "Recent thoughts and actions:\n"
    for entry in memory:
        step = entry.get('step', '?')
        thought = entry.get('thought', 'N/A')
        action = entry.get('action_name', entry.get('action', 'N/A'))
        summary += f"- Step {step}: Thought=[{thought}] -> Action=[{action}]\n"

    summary += "\nUse this history to avoid repeating the exact same actions if you are not making progress."
    return summary
