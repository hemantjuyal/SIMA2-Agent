"""
Agents Factory
"""
from gsima.utils import config
from .base import BaseAgent
from .gemini.world_model import WorldModelAgent 
from .jev.agent import JevAgent

def create_agent() -> BaseAgent:
    """
    Factory function to create the configured agent.
    """
    import logging
    backend = getattr(config, "RUNTIME", "gemini").lower()
    logging.info(f"========== INITIALIZING RUNTIME: {backend.upper()} ==========")
    if backend == "jev":
        return JevAgent(name="JevAgent", description="High-frequency typed agent.")
    return WorldModelAgent(name="WorldModelAgent", description="A world model agent that perceives, imagines, plans, and acts.")
