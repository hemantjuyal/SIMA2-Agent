from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    This class defines the high-level interface for an agent. The main
    orchestrator will instantiate a specific agent and call its `run` method.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, context: 'AgentContext'):
        """
        The primary entry point to execute the agent's logic.
        
        Args:
            context: An `AgentContext` object containing all necessary components
                     (environment, runtimes, memory, etc.) for the agent to operate.
        """
        pass
