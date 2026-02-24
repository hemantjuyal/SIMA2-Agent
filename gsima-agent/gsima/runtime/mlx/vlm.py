"""
VLM Runtime Adapter (Multimodal).

This module handles loading and inferencing with Vision-Language Models (VLM)
like LLaVA using MLX.
"""
import logging
import logging
import numpy as np
import mlx.core as mx
from mlx_vlm import load, generate
from PIL import Image
from typing import Optional # Import Optional for type hinting
from mlx_vlm.utils import get_model_path # Import to get model path for loading

from gsima.utils import config
from gsima.runtime.base import BaseModelRuntime # Import BaseModelRuntime

class MLXVLMRuntime(BaseModelRuntime):
    """
    VLM Runtime Adapter (Multimodal) for MLX.
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
        self.processor = None
        self.load_model()

    def load_model(self):
        """Loads the VLM specified by model_id."""
        if self.model is None:
            logging.info(f"Loading VLM model: {self.model_id}...")
            model_path = get_model_path(self.model_id)
            self.model, self.processor = load(model_path, processor_config={"use_fast": True})
            logging.info("VLM model loaded successfully.")

    def get_model_response(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        """
        Invokes the loaded VLM to get a textual response based on prompt and image.

        Args:
            prompt: The textual part of the prompt for the VLM.
            image: The numpy array representing the image (rgb_array from Gym).

        Returns:
            The textual response from the VLM.
        """
        if image is None:
            logging.error("VLM runtime requires an image but 'image' was None.")
            raise ValueError("Image observation is required for VLM runtime.")

        image_as_uint8 = image.astype(np.uint8)
        pil_image = Image.fromarray(image_as_uint8)
        
        inputs = self.processor(text=prompt, images=[pil_image])
        
        logging.debug("Generating response from VLM...")
        response_generator = generate(
            self.model,
            self.processor,
            inputs["pixel_values"],
            inputs["input_ids"],
            max_tokens=64,
            temp=0.0,
            verbose=False,
        )
        
        full_response = "".join(list(response_generator))
        
        logging.debug(f"VLM raw response: {full_response}")
        return full_response
