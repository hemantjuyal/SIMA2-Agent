"""
Base classes for model runtimes.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseModelRuntime(ABC):
    """Abstract Base Class for all model runtimes (LLM, VLM, etc.)."""

    @abstractmethod
    def get_model_response(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        """
        Invokes the model to get a textual response.
        Implementations should handle whether the 'image' argument is used.
        """
        pass
