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
    # In the new architecture the VLM is only asked to provide semantic labels
    # (e.g., triangle, keys, doors, lava); do NOT infer precise positions
    # or numeric state. Keep answers short and in a markdown list.
    return (
        "You are a semantic image interpreter for a grid-world. "
        "From the image, return only semantic facts that the simulator may not know. "
        "Do NOT attempt to report agent coordinates, orientation, or exact goal coordinates. "
        "Return a short markdown list with any of the following keys if present: Triangle, Square, Keys, Doors, Lava, Environment Description. "
        "Examples: '- **Triangle**: a red triangle at the top-left', '- **Square**: a green square at the bottom-right', '- **Keys**: 1 key visible at top-left', '- **Doors**: a closed wooden door at the east wall', '- **Lava**: lava tiles near the goal'."
    )


def get_controller_prompt(instruction: str, structured_perception: Dict[str, Any], memory_summary: str, imagined_futures: Dict[str, Dict[str, str]]) -> str:
    """
    Dynamically generates the main prompt for the Controller LLM, instructing it to plan
    before it acts by evaluating imagined futures.
    
    Args:
        instruction: The high-level instruction for the agent.
        structured_perception: A dictionary representing the perceived current state of the environment.
        memory_summary: A summarized string of the agent's recent experiences.
        imagined_futures: A dictionary where keys are actions and values are the predicted next states
                          if that action were taken.
    """
    action_list = [action.name for action in SUPPORTED_ACTIONS]
    
    # Format the perception dictionary as a simple key-value string for the prompt
    perception_items = [f"- {key.replace('_', ' ').title()}: {value}" for key, value in structured_perception.items()]
    perception_str = "\n".join(perception_items) if perception_items else "No visual data available."

    # Format imagined futures with obstacle highlighting
    imagined_futures_str = ""
    if imagined_futures:
        for action, predicted_state in imagined_futures.items():
            obstacle_flag = ""
            if predicted_state.get('obstacle_in_front', '').lower() == 'true':
                obstacle_flag = " ⚠️ WARNING: OBSTACLE AHEAD - AVOID THIS ACTION"
            imagined_futures_str += f"**If I choose action: {action}**{obstacle_flag}\n"
            for key, value in predicted_state.items():
                imagined_futures_str += f"- {key.replace('_', ' ').title()}: {value}\n"
            imagined_futures_str += "\n"
    else:
        imagined_futures_str = "No imagined futures available."

    return f"""You are an intelligent and methodical agent in a grid world. Your mission is to efficiently reach the green square.

**MISSION:** {instruction}

You must follow this process:
1.  **Analyze** your current situation from PERCEPTION and MEMORY.
2.  **IMAGINE** the possible outcomes of different actions (provided below as IMAGINED FUTURES).
3.  **PLAN** and formulate a clear, one-sentence rationale for your next move, considering the IMAGINED FUTURES and your MISSION.
4.  **ACT** by choosing a single action based on your plan.

---
**1. PERCEPTION (from Vision Model):**
{perception_str}

**2. MEMORY (Recent History):**
{memory_summary}

**3. IMAGINED FUTURES (Predicted by Internal Model):**
{imagined_futures_str}

---
**4. PLANNING AND ACTING:**
Based on your MISSION, PERCEPTION, MEMORY, and IMAGINED FUTURES, provide your thought process and the single best action to take right now.

**CRITICAL DECISION RULES:**
1. **NEVER choose an action where "Obstacle In Front: true"** - This will result in collision and wasted moves.
2. **Prioritize actions where "Obstacle In Front: false"** - These are the only valid movement options.
3. **Evaluate goal position**: Choose actions that orient you toward or move you closer to the green square.
4. **Do NOT repeat ineffective actions** - Learn from memory that previous similar moves failed.
5. **Do NOT simply repeat thoughts from previous steps** - Every step is a new situation.

Return your decision in a markdown list format.
- The "thought" value MUST be your unique one-sentence rationale for THIS step, explaining why you avoided obstacles and chose this action.
- The "action" value MUST be one of: {action_list}

Example (DO NOT COPY THIS TEXT):
- **thought**: [Obstacle ahead if I move forward, but turning right shows clear path with goal to my left, so I will turn right to align toward goal.]
- **action**: [TURN_RIGHT]

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


def choose_safe_action(imagined_futures: Dict[str, Dict[str, str]], fallback_action: str) -> str:
    """Choose the safest action based on MiniGrid-specific goal-relative semantics."""
    def _score_goal_alignment(goal_relative_position: str) -> int:
        normalized = goal_relative_position.strip().lower()
        ranking = {
            'here': 0,
            'front': 1,
            'front-left': 2,
            'front-right': 2,
            'left': 3,
            'right': 3,
            'back-left': 4,
            'back-right': 4,
            'back': 5,
        }
        return ranking.get(normalized, 10)

    safe_actions = [
        (action, _score_goal_alignment(future.get('goal_relative_position', '')))
        for action, future in imagined_futures.items()
        if future.get('obstacle_in_front', '').lower() != 'true' and action != 'STOP'
    ]
    if not safe_actions:
        if imagined_futures.get('STOP', {}).get('obstacle_in_front', '').lower() != 'true':
            return 'STOP'
        return fallback_action

    safe_actions.sort(key=lambda item: item[1])
    return safe_actions[0][0]

def create_memory_summary(memory: Deque[Dict[str, Any]]) -> str:
    """
    Creates a summarized string of the agent's recent memory, including thoughts
    and quantitative rewards.
    """
    if not memory:
        return "No history yet."

    summary = "Recent structured transitions:\n"

    for entry in memory:
        step = entry.get('step', '?')
        action = entry.get('action_name', entry.get('action', 'N/A'))
        reward = entry.get('reward', None)
        dist = entry.get('distance_to_goal', None)
        pos = entry.get('agent_pos', None)

        reward_str = f"{reward:.2f}" if reward is not None else "N/A"
        dist_str = str(dist) if dist is not None else "N/A"
        pos_str = str(pos) if pos is not None else "N/A"

        summary += f"- Step {step}: action={action} | pos={pos_str} | dist_to_goal={dist_str} | reward={reward_str}\n"

    summary += "Use these recent transitions to avoid repeating ineffective actions and to favor trajectories that reduced distance to goal."
    return summary
