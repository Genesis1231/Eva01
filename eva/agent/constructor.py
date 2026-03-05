"""
PromptConstructor:
    Constructs and formats prompts for the chat agent.
    Assembles various components into structured prompts by combining:
    - Persona and instruction templates loaded from files
    - Conversation history
    - observations and inputs
"""

from config import logger
from pydantic import BaseModel, Field
from eva.utils.prompt import load_prompt

class PromptConstructor:
    """
    PromptConstructor class:
    
    Attributes:
        persona_prompt (str): The persona prompt loaded from the persona.md file.
        instruction_prompt (str): The instruction prompt loaded from the instructions.md file.
    """
    
    def __init__(self):
        self.soul: str = load_prompt("SOUL") # default persona prompt
        self.instructions: str = load_prompt("INSTRUCTIONS") # default instructions prompt
        

    def build_system(self, timestamp: str, memory: str = None) -> str:
        """Build the system prompt string."""
        prompt = (
            f"<PERSONA>{self.soul}</PERSONA>\n\n"
            f"<INSTRUCTIONS>\n"
            f"{self.instructions}\n"
            f"</INSTRUCTIONS>"        
        )
        if memory:
            prompt += f"\n\n<MEMORY>\n{memory}\n</MEMORY>"
        
        prompt += f"\n\n<CURRENT_TIME>{timestamp}</CURRENT_TIME>\n\n"
        return prompt
