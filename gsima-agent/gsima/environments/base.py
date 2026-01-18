"""Base classes for environment and adapter definitions."""
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    """Abstract base class for all environment adapters."""
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def translate_action(self, action_name: str) -> any:
        """Translates a canonical action name into an env-specific action."""
        pass
