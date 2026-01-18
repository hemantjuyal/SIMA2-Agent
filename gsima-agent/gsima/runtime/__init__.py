"""
Runtime Factory

This module dynamically loads and initializes the correct model runtimes
based on the configuration.
"""
import logging
from gsima.utils import config

def get_vlm_runtime():
    """
    Factory function to get the VLM runtime instance based on config.
    """
    runtime_type = config.RUNTIME.lower()
    
    if runtime_type == "ollama":
        try:
            from gsima.runtime.ollama.vlm import get_ollama_vlm_runtime
            logging.info("Loading VLM runtime (Ollama)...")
            return get_ollama_vlm_runtime()
        except ImportError as e:
            logging.error("Could not import Ollama VLM runtime. Please ensure Ollama is installed and configured.")
            raise e
    elif runtime_type == "mlx":
        try:
            from gsima.runtime.mlx.vlm import get_vlm_runtime
            logging.info("Loading VLM runtime (MLX)...")
            return get_vlm_runtime()
        except ImportError as e:
            logging.error("Could not import MLX VLM runtime. Please ensure MLX is installed.")
            raise e
    else:
        raise ValueError(f"Unsupported RUNTIME for VLM: '{runtime_type}'. Must be 'mlx' or 'ollama'.")

def get_llm_runtime():
    """
    Factory function to get the LLM runtime instance based on config.
    """
    runtime_type = config.RUNTIME.lower()
    
    if runtime_type == "ollama":
        try:
            from gsima.runtime.ollama.llm import get_ollama_llm_runtime
            logging.info("Loading LLM runtime (Ollama)...")
            return get_ollama_llm_runtime()
        except ImportError as e:
            logging.error("Could not import Ollama LLM runtime. Please ensure Ollama is installed and configured.")
            raise e
    elif runtime_type == "mlx":
        try:
            from gsima.runtime.mlx.llm import get_llm_runtime
            logging.info("Loading LLM runtime (MLX)...")
            return get_llm_runtime()
        except ImportError as e:
            logging.error("Could not import MLX LLM runtime. Please ensure MLX is installed.")
            raise e
    else:
        raise ValueError(f"Unsupported RUNTIME for LLM: '{runtime_type}'. Must be 'mlx' or 'ollama'.")