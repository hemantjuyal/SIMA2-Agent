"""
Ollama LLM Runtime Adapter.

This module handles inferencing with Language Models (LLM)
served by an Ollama instance.
"""
import logging
import requests
from typing import Optional

from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime

class OllamaLLMRuntime(BaseModelRuntime):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OllamaLLMRuntime, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        """
        Checks if the Ollama service is running. This function doesn't load a model
        into memory but verifies connectivity to the Ollama server.
        """
        logging.info(f"Checking connection to Ollama server at {config.OLLAMA_BASE_URL}...")
        try:
            response = requests.get(config.OLLAMA_BASE_URL, timeout=5)
            if response.status_code == 200:
                logging.info("Ollama server is running.")
            else:
                logging.warning(f"Ollama server returned status code {response.status_code}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to connect to Ollama server: {e}")
            raise RuntimeError(f"Could not connect to Ollama server at {config.OLLAMA_BASE_URL}")

    def get_model_response(self, prompt: str, image: Optional[bytes] = None) -> str:
        """
        Invokes the Ollama API to get a textual response based on a prompt.

        Args:
            prompt: The textual part of the prompt for the LLM.
            image: This is not used by the LLM runtime.

        Returns:
            The textual response from the LLM.
        """
        try:
            # Construct the payload for the Ollama API
            payload = {
                "model": config.OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False # Get the full response at once
            }

            api_url = f"{config.OLLAMA_BASE_URL}/api/generate"
            logging.debug(f"Sending request to Ollama API at {api_url} with model {config.OLLAMA_LLM_MODEL}")
            
            response = requests.post(api_url, json=payload, timeout=500)
            response.raise_for_status() # Raise an exception for bad status codes

            response_data = response.json()
            full_response = response_data.get("response", "").strip()

            logging.debug(f"Ollama LLM raw response: {full_response}")
            return full_response

        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama API request failed: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during Ollama LLM call: {e}")
            raise RuntimeError(f"Unexpected error in Ollama LLM runtime: {e}")

def get_ollama_llm_runtime():
    """Returns the singleton instance of the OllamaLLMRuntime."""
    runtime = OllamaLLMRuntime()
    runtime.load_model()
    return runtime
