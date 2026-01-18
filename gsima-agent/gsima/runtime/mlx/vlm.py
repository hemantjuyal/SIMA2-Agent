"""
VLM Runtime Adapter (Multimodal).

This module handles loading and inferencing with Vision-Language Models (VLM)
like LLaVA using MLX.
"""
import logging
import numpy as np
import mlx.core as mx
from mlx_vlm import load, generate
from PIL import Image
from typing import Optional # Import Optional for type hinting
from mlx_vlm.utils import get_model_path # Import to get model path for loading

from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime # Import BaseModelRuntime

class VLMRuntime(BaseModelRuntime): # Inherit from BaseModelRuntime
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLMRuntime, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.processor = None # For LLaVA, this is the image processor
        return cls._instance

    def load_model(self):
        """Loads the VLM specified in the config."""
        if self.model is None:
            logging.info(f"Loading VLM model: {config.VLM_MODEL_ID}...")
            # For mlx_vlm, load returns model and processor
            # mlx_vlm.load expects `get_model_path(model_name)` to resolve local/HF path
            model_path = get_model_path(config.VLM_MODEL_ID)
            self.model, self.processor = load(model_path, processor_config={"use_fast": True})
            logging.info("VLM model loaded successfully.")

    def get_model_response(self, prompt: str, image: Optional[np.ndarray] = None) -> str: # Conforming signature
        """
        Invokes the loaded VLM to get a textual response based on prompt and image.

        Args:
            prompt: The textual part of the prompt for the VLM.
            image: The numpy array representing the image (rgb_array from Gym).

        Returns:
            The textual response from the VLM.
        """
        if self.model is None:
            self.load_model()
        
        if image is None:
            logging.error("VLM runtime requires an image but 'image' was None.")
            raise ValueError("Image observation is required for VLM runtime.")

        # Convert numpy array image to PIL Image, ensuring correct data type
        image_as_uint8 = image.astype(np.uint8)
        pil_image = Image.fromarray(image_as_uint8)
        
        # Processor tokenizes the text prompt and prepares the image
        inputs = self.processor(text=prompt, images=[pil_image])
        
        # The processor is expected to return MLX arrays directly
        # No need to explicitly convert here: inputs = {k: mx.array(v) for k, v in inputs.items()}

        logging.debug("Generating response from VLM...")
        # Correct mlx_vlm.generate call: takes image_pixels and input_ids explicitly
        response_generator = generate(
            self.model,
            self.processor, # processor is needed here
            inputs["pixel_values"], # This is what mlx_vlm.generate expects for image data
            inputs["input_ids"],    # This is what mlx_vlm.generate expects for tokenized prompt
            max_tokens=64, # Max tokens for the VLM's description/response
            temp=0.0, # Explicitly set temperature for deterministic output
            verbose=False,  # Set to False to avoid excessive output
        )
        
        # mlx_vlm.generate returns a generator for streaming output; concatenate for full response
        full_response = "".join(list(response_generator))
        
        logging.debug(f"VLM raw response: {full_response}")
        return full_response

def get_vlm_runtime():
    """Returns the singleton instance of the VLMRuntime."""
    runtime = VLMRuntime()
    runtime.load_model()
    return runtime
