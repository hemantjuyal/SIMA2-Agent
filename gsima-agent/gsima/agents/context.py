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
    that an agent needs to operate.
    """
    env: gym.Env
    adapter: BaseAdapter
    multimodal_runtime: BaseModelRuntime 
    memory_system: BaseMemory
    
    get_multimodal_prompt: Callable
    event_emitter: Callable = None
