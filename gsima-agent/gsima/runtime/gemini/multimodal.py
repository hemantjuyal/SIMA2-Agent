import logging
import os
from PIL import Image
from typing import Any
from google import genai
from gsima.runtime.base import BaseModelRuntime

class GeminiMultimodalRuntime(BaseModelRuntime):
    """
    Unified Multimodal Runtime using Google Gemini (google-genai).
    Handles both image and text inputs in a single call.
    """
    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini runtime.")
        self.client = genai.Client(api_key=api_key)

    def get_model_response(self, prompt: str, image: Any = None) -> str:
        """
        Invokes Gemini with the given prompt and optional image.
        Image is expected to be a numpy array (rgb_array) from Gym.
        """
        contents = []
        if image is not None:
            # Convert rgb_array (numpy) to PIL Image
            try:
                pil_img = Image.fromarray(image)
                contents.append(pil_img)
            except Exception as e:
                logging.warning(f"Failed to convert image array to PIL Image: {e}")
            
        contents.append(prompt)
        
        logging.info(f"Sending multimodal request to Gemini model: {self.model_id}")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini API request failed: {e}")
            raise RuntimeError(f"Gemini API request failed: {e}")
