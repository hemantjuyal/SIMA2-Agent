import logging
from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime

def create_runtime(runtime_name: str, model_type: str, model_id: str) -> BaseModelRuntime:
    """
    Factory function to create a runtime instance based on configuration.
    """
    runtime_name = runtime_name.lower()
    model_type = model_type.lower()

    if runtime_name == "gemini":
        if model_type == "multimodal":
            from gsima.runtime.gemini.multimodal import GeminiMultimodalRuntime
            logging.info(f"Loading Gemini Multimodal runtime with model: {model_id}...")
            return GeminiMultimodalRuntime(model_id)
        else:
            raise ValueError(f"Unsupported model type for gemini: '{model_type}'. Must be 'multimodal'.")
    else:
        raise ValueError(f"Unsupported RUNTIME: '{runtime_name}'. Must be 'gemini'.")