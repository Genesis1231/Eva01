import numpy as np
import base64
import cv2
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from config import logger
from eva.utils.prompt import load_prompt


class Describer:
    """EVA's visual cortex — preprocesses images and delegates to a vision model."""

    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model = init_chat_model(model_name, temperature=temperature)
        logger.debug(f"Describer: {model_name} is ready.")

    def _convert_base64(self, image_data: np.ndarray | str) -> str:
        """ Convert image data to base64 string if it's a numpy array."""
        if isinstance(image_data, str):
            return image_data
        _, buffer = cv2.imencode('.jpg', image_data)
        
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    async def _generate(self, image_base64: str, prompt: str) -> str | None:
        """ Invoke the vision model"""
        
        if not image_base64:
            logger.error("Describer: No image data provided.")
            return None
        
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            },
        ])
        
        try:
            response = await self.model.ainvoke([message])
            return str(response.content) 
        
        except Exception as e:
            logger.error(f"Describer: model invocation failed — {e}")
            return None
        
    async def describe(self, image_data: np.ndarray | str) -> str | None:
        try:
            image_base64 = self._convert_base64(image_data)
            prompt = load_prompt("vision")  
        
            return await self._generate(image_base64, prompt)
        except Exception as e:
            logger.error(f"Describer: failed — {e}")
            return None

    async def analyze_screenshot(self, image_data: np.ndarray | str, query: str) -> str | None:
        try:
            image_base64 = self._convert_base64(image_data)
            prompt = f"Analyze the screenshot with respect to this question: {query}."
            
            return await self._generate(image_base64, prompt)
        except Exception as e:
            logger.error(f"Describer: screenshot failed — {e}")
            return None
