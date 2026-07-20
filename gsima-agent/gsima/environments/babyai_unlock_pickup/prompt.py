"""Prompts and summaries for BabyAI UnlockPickup environments."""
from typing import Deque, Dict, Any
from gsima.environments.babyai_unlock_pickup.schema import SUPPORTED_ACTIONS

def get_multimodal_prompt(instruction: str, memory_summary: str) -> str:
    """Create a multimodal prompt for the BabyAI action selection loop."""
    action_list = ", ".join([a.name for a in SUPPORTED_ACTIONS])
    
    return f"""You are an agent controlling a grid-world environment.
Your goal is to complete the mission based on the visual input.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** the image provided (your current visual perception).
2.  **Read** your MEMORY to avoid repeating mistakes or loops.
3.  **Think** about the best move to achieve the mission.
4.  **Act** by choosing a single action.

---
**MEMORY (Recent History):**
{memory_summary}

---
**DECISION RULES:**
- You can only see the grid tiles directly in front of you.
- To explore, use TURN_LEFT, TURN_RIGHT, and MOVE_FORWARD to move around.
- **COLOR MATCHING IS CRITICAL:** There may be multiple keys, doors, or boxes of different colors. You MUST strictly match the color of the object you interact with to the color specified in your MISSION.
- If you see the correct color key and you are not holding anything, navigate to it until it is in the tile directly in front of you, then use PICK_UP.
- If you are carrying the key and see the correct color closed door, navigate to the door until it is directly in front of you, then use TOGGLE to unlock and open it.
- Once the door is open, enter the room to find the target object.
- If you see the target object, navigate to it until it is directly in front of you, then use PICK_UP. (You may need to DROP the key first if your hands are full).
- **IMPORTANT DROP RULE:** You CANNOT drop an object if the tile directly in front of you is occupied by another object, a door, or a wall. You must turn to face an EMPTY floor tile first before using DROP.
- Only use PICK_UP, DROP, and TOGGLE when you are directly facing the target object or door (1 tile away).

Return your decision in a strict markdown list format with these exactly 4 keys:
- **perception**: [Briefly describe the exact visual state of the environment, including the colors of any objects or doors]
- **thought**: [Based strictly on the perception, state your strategic rationale for the next move]
- **narrative**: [Based on the thought, write an engaging, first-person sentence declaring your action. Example: "I am dropping the green key so my hands are free to pick up the purple box."]
- **action**: [The exact action that executes the thought. Must be exactly one of: {action_list}]
"""

def create_memory_summary(memory_deque: Deque[Dict[str, Any]]) -> str:
    """Summarize recent memory for the prompt."""
    if not memory_deque:
        return "No history yet."
    
    summary_lines = []
    for entry in list(memory_deque)[-5:]:
        action = entry.get("action_name", entry.get("action", "UNKNOWN"))
        thought = entry.get("thought", "No thought recorded.")
        summary_lines.append(f"- Chose {action} because: {thought}")
        
    return "\n".join(summary_lines)
