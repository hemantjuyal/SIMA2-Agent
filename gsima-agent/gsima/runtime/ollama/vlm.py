"""
Ollama VLM Runtime Adapter.

This module handles inferencing with Vision-Language Models (VLM)
served by an Ollama instance.
"""
import logging
import logging
import requests
import base64
import io
import numpy as np
from PIL import Image
from typing import Optional

from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime

class OllamaVLMRuntime(BaseModelRuntime):
    """
    Handles inferencing with Vision-Language Models (VLM) served by an Ollama instance.
    This is a non-singleton class, allowing multiple instances with different models.
    """
    def __init__(self, model_id: str):
        """
        Initializes the runtime for a specific Ollama model.

        Args:
            model_id: The name of the Ollama model to use (e.g., 'llava:latest').
        """
        self.model_id = model_id
        self.base_url = config.OLLAMA_BASE_URL
        self._check_connection()


    def _check_connection(self):
        """
        Checks if the Ollama service is running. This function doesn't load a model
        into memory but verifies connectivity to the Ollama server.
        """
        logging.info(f"Checking connection to Ollama server at {self.base_url}...")
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                logging.info("Ollama server is running.")
            else:
                logging.warning(f"Ollama server returned status code {response.status_code}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to connect to Ollama server: {e}")
            raise RuntimeError(f"Could not connect to Ollama server at {self.base_url}")

    def get_model_response(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        """
        Invokes the Ollama API to get a textual response based on prompt and image.

        Args:
            prompt: The textual part of the prompt for the VLM.
            image: The numpy array representing the image (rgb_array from Gym).

        Returns:
            The textual response from the VLM.
        """
        if image is None:
            logging.error("Ollama VLM runtime requires an image but 'image' was None.")
            raise ValueError("Image observation is required for VLM runtime.")

        try:
            # Convert numpy array to base64 string
            pil_image = Image.fromarray(image.astype(np.uint8))
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Construct the payload for the Ollama API
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False, # Get the full response at once
                "options": {
                    "num_ctx": config.OLLAMA_CONTEXT_SIZE
                }
            }

            api_url = f"{self.base_url}/api/generate"
            logging.debug(f"Sending request to Ollama API at {api_url} with model {self.model_id}")
            
            response = requests.post(api_url, json=payload, timeout=config.OLLAMA_REQUEST_TIMEOUT)
            response.raise_for_status() # Raise an exception for bad status codes

            response_data = response.json()
            full_response = response_data.get("response", "").strip()

            logging.debug(f"Ollama VLM raw response: {full_response}")
            return full_response

        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama API request failed: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during Ollama VLM call: {e}")
            raise RuntimeError(f"Unexpected error in Ollama VLM runtime: {e}")
