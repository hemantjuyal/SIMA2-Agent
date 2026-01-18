from collections import deque
from typing import Any, Dict, List

from gsima.memory.base import BaseMemory
from gsima.utils import config

class DequeMemory(BaseMemory):
    """
    A simple short-term memory implementation using a `collections.deque`.

    This memory system stores a fixed number of the most recent experiences
    and relies on an external 'summarizer' function to format its content
    for the LLM prompt.
    """

    def __init__(self):
        """Initializes the deque with a maximum length from the config."""
        self.buffer = deque(maxlen=config.MEMORY_LENGTH)
        self.summarizer = None

    def set_summarizer(self, summarizer_func: callable):
        """
        Sets the environment-specific function used to summarize the memory
        buffer for the prompt.
        """
        self.summarizer = summarizer_func

    def add(self, experience: Dict[str, Any]):
        """Adds a new experience dictionary to the buffer."""
        self.buffer.append(experience)

    def retrieve(self) -> str:
        """
        Retrieves the memory as a summarized string by calling the
        injected summarizer function.
        """
        if not self.summarizer:
            raise ValueError(
                "Memory summarizer has not been set. "
                "The environment needs to provide a summarizer function via `set_summarizer()`."
            )
        return self.summarizer(self.buffer)

    def clear(self):
        """Clears all experiences from the memory buffer."""
        self.buffer.clear()
