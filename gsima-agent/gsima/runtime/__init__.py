import logging
from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime

def create_runtime(runtime_name: str, model_type: str, model_id: str) -> BaseModelRuntime:
    """
    Factory function to create a runtime instance based on configuration.
    """
    runtime_name = runtime_name.lower()
    model_type = model_type.lower()

    if runtime_name == "ollama":
        if model_type == "llm":
            try:
                from gsima.runtime.ollama.llm import OllamaLLMRuntime
                logging.info(f"Loading Ollama LLM runtime with model: {model_id}...")
                return OllamaLLMRuntime(model_id)
            except ImportError as e:
                logging.error("Could not import Ollama LLM runtime. Please ensure Ollama is installed and configured.")
                raise e
        elif model_type == "vlm":
            try:
                from gsima.runtime.ollama.vlm import OllamaVLMRuntime
                logging.info(f"Loading Ollama VLM runtime with model: {model_id}...")
                return OllamaVLMRuntime(model_id)
            except ImportError as e:
                logging.error("Could not import Ollama VLM runtime. Please ensure Ollama is installed and configured.")
                raise e
        else:
            raise ValueError(f"Unsupported model type for Ollama: '{model_type}'. Must be 'llm' or 'vlm'.")
    elif runtime_name == "mlx":
        if model_type == "llm":
            try:
                from gsima.runtime.mlx.llm import MLXLLMRuntime
                logging.info(f"Loading MLX LLM runtime with model: {model_id}...")
                return MLXLLMRuntime(model_id)
            except ImportError as e:
                logging.error("Could not import MLX LLM runtime. Please ensure MLX is installed.")
                raise e
        elif model_type == "vlm":
            try:
                from gsima.runtime.mlx.vlm import MLXVLMRuntime
                logging.info(f"Loading MLX VLM runtime with model: {model_id}...")
                return MLXVLMRuntime(model_id)
            except ImportError as e:
                logging.error("Could not import MLX VLM runtime. Please ensure MLX is installed.")
                raise e
        else:
            raise ValueError(f"Unsupported model type for MLX: '{model_type}'. Must be 'llm' or 'vlm'.")
    else:
        raise ValueError(f"Unsupported RUNTIME: '{runtime_name}'. Must be 'mlx' or 'ollama'.")