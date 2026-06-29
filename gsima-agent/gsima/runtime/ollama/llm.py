"""
Ollama LLM Runtime Adapter.

This module handles inferencing with Language Models (LLM)
served by an Ollama instance.
"""
import logging
import logging
import requests
from typing import Optional

from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime

class OllamaLLMRuntime(BaseModelRuntime):
    """
    Handles inferencing with Language Models (LLM) served by an Ollama instance.
    This is a non-singleton class, allowing multiple instances with different models.
    """

    def __init__(self, model_id: str):
        """
        Initializes the runtime for a specific Ollama model.

        Args:
            model_id: The name of the Ollama model to use (e.g., 'qwen2:1.5b').
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
            # First attempt: OpenAI-compatible endpoint.
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.0,
            }

            api_url_v1 = f"{self.base_url}/v1/completions"
            logging.debug(f"Sending request to Ollama API at {api_url_v1} with model {self.model_id}")

            try:
                response = requests.post(api_url_v1, json=payload, timeout=config.OLLAMA_REQUEST_TIMEOUT)
                response.raise_for_status()
                response_data = response.json()
                choices = response_data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"No choices returned from Ollama LLM response: {response_data}")
                return choices[0].get("text", "").strip()
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, 'status_code', None)
                if status == 404:
                    logging.warning(f"Ollama v1/completions endpoint not found, falling back to /api/generate: {e}")
                else:
                    logging.warning(f"Ollama v1/completions request failed, falling back to /api/generate: {e}")
            except requests.exceptions.RequestException as e:
                logging.warning(f"Ollama v1/completions request error, falling back to /api/generate: {e}")

            # Fallback: legacy /api/generate endpoint
            payload_api = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": config.OLLAMA_CONTROLLER_CONTEXT_SIZE
                }
            }
            api_url_api = f"{self.base_url}/api/generate"
            logging.debug(f"Sending fallback request to Ollama API at {api_url_api} with model {self.model_id}")
            response = requests.post(api_url_api, json=payload_api, timeout=config.OLLAMA_REQUEST_TIMEOUT)
            if response.status_code == 404:
                # Try fallback model if available
                fallback_model = getattr(config, 'OLLAMA_CONTROLLER_FALLBACK_MODEL', None)
                if fallback_model and fallback_model != self.model_id:
                    logging.warning(f"Model '{self.model_id}' not found on /api/generate; retrying with fallback model '{fallback_model}'")
                    payload_api['model'] = fallback_model
                    response = requests.post(api_url_api, json=payload_api, timeout=config.OLLAMA_REQUEST_TIMEOUT)
            response.raise_for_status()
            response_data = response.json()
            full_response = response_data.get("response", "").strip()
            if not full_response:
                raise RuntimeError(f"Empty response from Ollama /api/generate for model {self.model_id}")
            return full_response

            logging.debug(f"Ollama LLM raw response: {full_response}")
            return full_response

        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama API request failed: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during Ollama LLM call: {e}")
            raise RuntimeError(f"Unexpected error in Ollama LLM runtime: {e}")
