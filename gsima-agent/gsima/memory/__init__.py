"""
Memory System Factory

This module provides a factory function to instantiate the memory system
specified in the project configuration.
"""
from gsima.utils import config
from .short_term import DequeMemory
from .base import BaseMemory

def create_memory() -> BaseMemory:
    """
    Factory function to create the configured memory system.

    Currently, it defaults to `DequeMemory`. This can be expanded to read a
    `config.MEMORY_TYPE` variable to switch between different memory
    implementations (e.g., 'short_term', 'vector_db', 'entity_graph').
    """
    # memory_type = config.MEMORY_TYPE.lower()
    # if memory_type == 'vector_db':
    #     return VectorDBMemory()
    # else:
    #     return DequeMemory()
    
    return DequeMemory()
