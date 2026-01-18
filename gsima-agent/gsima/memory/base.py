from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseMemory(ABC):
    """
    Abstract base class for all memory systems. This defines the contract
    that the agent's `run_agent_loop` will use to interact with any memory module.
    """

    @abstractmethod
    def add(self, experience: Dict[str, Any]):
        """Adds a new experience to the memory.

        An 'experience' is a dictionary containing all relevant information
        from a single timestep (e.g., observation, action, reward, outcome).
        """
        pass

    @abstractmethod
    def retrieve(self) -> str:
        """
        Retrieves and formats the memory for use in the agent's prompt.
        
        The format of the returned value (e.g., a string summary, a list of
        examples) depends on the implementation and the summarizer it uses.
        """
        pass

    @abstractmethod
    def set_summarizer(self, summarizer_func: callable):
        """
        Provides the memory system with the environment-specific tool to format
        its content into a string suitable for the LLM prompt.
        """
        pass

    @abstractmethod
    def clear(self):
        """Clears the memory, such as at the start of a new episode."""
        pass
