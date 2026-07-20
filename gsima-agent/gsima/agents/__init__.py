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
    return WorldModelAgent(name="WorldModelAgent", description="A world model agent that perceives, imagines, plans, and acts.")
