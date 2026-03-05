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
        self.persona_prompt: str = load_prompt("SOUL") # default persona prompt
        self.instruction: str = load_prompt("PERSONA") # default instructions prompt
        

    def build_system(self, timestamp: str, memory: str = None) -> str:
        """Build the system prompt string."""
        prompt = (
            f"<PERSONA>{self.persona_prompt}</PERSONA>\n\n"
            f"<CURRENT_TIME>{timestamp}</CURRENT_TIME>\n\n"
            f"<INSTRUCTIONS>\n"
            f"{self.instruction}\n"
            f"</INSTRUCTIONS>"
        )
        if memory:
            prompt += f"\n\n<MEMORY>\n{memory}\n</MEMORY>"
        return prompt

    def build_prompt(
        self,
        timestamp: str,
        sense: str,
    ) -> tuple[str, str]:
        """Build system and human messages. Returns (system, human)."""

        system_prompt = self.build_system(timestamp)

        human_prompt = (
            f"{sense}\n"
        )

        logger.debug(f"PromptConstructor: system — \n{system_prompt}")
        logger.debug(f"PromptConstructor: human — \n{human_prompt}")
        return system_prompt, human_prompt
