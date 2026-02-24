"""
LLM Runtime Adapter (Text-only).
"""
import logging
import logging
from typing import Optional
import numpy as np # For type hinting np.ndarray
from mlx_lm import load, generate
from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime

class MLXLLMRuntime(BaseModelRuntime):
    """
    LLM Runtime Adapter (Text-only).
    This is a non-singleton class, allowing multiple instances with different models.
    """
    def __init__(self, model_id: str):
        """
        Initializes the runtime and loads the specified model.

        Args:
            model_id: The Hugging Face repository ID of the model to load.
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Loads the model specified in the config."""
        if self.model is None:
            logging.info(f"Loading LLM model: {self.model_id}...")
            try:
                self.model, self.tokenizer = load(self.model_id)
                logging.info("LLM model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load LLM model '{self.model_id}': {e}")
                raise RuntimeError(f"Failed to load LLM model: {e}")

    def get_model_response(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        """Invokes the loaded LLM via MLX to get a response."""
        if image is not None:
            logging.warning("Image provided to text-only LLM runtime. Image will be ignored.")

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=256, # Increased for potentially more complex outputs
                verbose=False,
            )
            return response
        except Exception as e:
            logging.error(f"Failed to generate response from LLM: {e}")
            raise RuntimeError(f"Failed to generate LLM response: {e}")
