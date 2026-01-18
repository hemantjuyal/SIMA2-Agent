from dataclasses import dataclass
import gymnasium as gym
from typing import Callable

from gsima.environments.base import BaseAdapter
from gsima.runtime.base import BaseModelRuntime
from gsima.memory.base import BaseMemory

@dataclass
class AgentContext:
    """
    A dataclass to hold all the necessary components and functions
    that an agent needs to operate. This serves as a dependency container
    for the agent's `run` method.
    """
    env: gym.Env
    adapter: BaseAdapter
    vlm_runtime: BaseModelRuntime # Can be None if AGENT_MODEL_TYPE is llm
    llm_runtime: BaseModelRuntime
    memory_system: BaseMemory
    
    # Environment-specific functions
    get_prompt: Callable
    get_visual_prompt: Callable
    get_outcome_from_reward: Callable
