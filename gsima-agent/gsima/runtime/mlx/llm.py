"""
LLM Runtime Adapter (Text-only).
"""
import logging
from typing import Optional
import numpy as np # For type hinting np.ndarray
from mlx_lm import load, generate
from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime

class MLXRuntime(BaseModelRuntime): # Inherit from BaseModelRuntime
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLXRuntime, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance

    def load_model(self):
        """Loads the model specified in the config."""
        if self.model is None:
            logging.info(f"Loading LLM model: {config.LLM_MODEL_ID}...")
            try:
                self.model, self.tokenizer = load(config.LLM_MODEL_ID)
                logging.info("LLM model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load LLM model '{config.LLM_MODEL_ID}': {e}")
                raise RuntimeError(f"Failed to load LLM model: {e}")

    def get_model_response(self, prompt: str, image: Optional[np.ndarray] = None) -> str: # Conforming signature
        """Invokes the loaded LLM via MLX to get a JSON response."""
        if self.model is None:
            self.load_model()
            
        if image is not None:
            logging.warning("Image provided to text-only LLM runtime. Image will be ignored.")

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=64,
                verbose=False, # Set to False to avoid excessive output
            )
            return response
        except Exception as e:
            logging.error(f"Failed to generate response from LLM: {e}")
            raise RuntimeError(f"Failed to generate LLM response: {e}")

def get_llm_runtime():
    """Returns the singleton instance of the MLXRuntime."""
    runtime = MLXRuntime()
    runtime.load_model()
    return runtime
