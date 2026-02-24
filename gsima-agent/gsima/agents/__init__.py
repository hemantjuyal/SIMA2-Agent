"""
Agents Factory
"""
from gsima.utils import config
from .base import BaseAgent
from .world_model import WorldModelAgent # Importing the WorldModelAgent

def create_agent() -> BaseAgent:
    """
    Factory function to create the configured agent.
    """
    agent_arch = config.AGENT_ARCH.lower()
    if agent_arch == 'world_model':
        return WorldModelAgent(name="WorldModelAgent", description="A world model agent that perceives, imagines, plans, and acts.")
    else:
        raise ValueError(f"Unsupported agent architecture: {agent_arch}. Only 'world_model' is supported currently.")
