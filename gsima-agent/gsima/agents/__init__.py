"""
Agents Factory
"""
from gsima.utils import config
from .base import BaseAgent
from .simple_loop import SimpleLoopAgent

def create_agent() -> BaseAgent:
    """
    Factory function to create the configured agent.
    
    This can be expanded to read a `config.AGENT_TYPE` to switch between
    different agent architectures (e.g., 'simple_loop', 'planner', 'hierarchical').
    """
    # agent_type = config.AGENT_TYPE.lower()
    # if agent_type == 'planner':
    #     return PlanningAgent()
    
    # Default to the simple loop agent for now
    return SimpleLoopAgent(name="SimpleLoopAgent", description="A reactive agent that operates in a step-by-step loop.")
