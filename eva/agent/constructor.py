"""
PromptConstructor:
    Constructs and formats prompts for the chat agent.
    Assembles various components into structured prompts by combining:
    - Persona and instruction templates loaded from files
    - Conversation history
    - observations and inputs
"""

from config import logger
from eva.utils.prompt import load_prompt
from eva.core.people import PeopleDB

class PromptConstructor:
    """
    PromptConstructor class:

    Attributes:
        persona_prompt (str): The persona prompt loaded from the persona.md file.
        instruction_prompt (str): The instruction prompt loaded from the instructions.md file.
    """

    def __init__(self, people_db: PeopleDB):
        self.soul: str = load_prompt("SOUL") # default persona prompt
        self.instructions: str = load_prompt("INSTRUCTIONS") # default instructions prompt
        self.people_db = people_db

    def build_system(
        self, 
        timestamp: str, 
        memory: str = "", 
        present_people: list[str] = []
        ) -> str:
        """Build the system prompt string."""
        
        prompt = (
            f"<PERSONA>{self.soul}</PERSONA>\n\n"
            f"<INSTRUCTIONS>\n"
            f"{self.instructions}\n"
            f"</INSTRUCTIONS>"
        )

        people_block = self._build_people_block(present_people)
        has_memory = people_block or memory

        if has_memory:
            prompt += "\n\n<MEMORY>"
            if people_block:
                prompt += f"\n{people_block}"
            if memory:
                prompt += f"\n{memory}"
            prompt += "\n</MEMORY>"

        prompt += f"\n\n<CURRENT_TIME>{timestamp}</CURRENT_TIME>\n\n"
        
        logger.debug(f"Constructed system prompt:\n{prompt}")
        return prompt

    def _build_people_block(self, present_people: list[str] | None) -> str :
        """Build <PEOPLE> block from face IDs currently visible to EVA."""
        
        if not present_people or not self.people_db:
            return ""

        entries = []
        for person_id in present_people:
            person = self.people_db.get(person_id)
            if not person:
                continue

            name = person["name"]
            rel = person.get("relationship") or "unknown"
            notes = person.get("notes") or ""

            entry = f"{name} ({rel})"
            if notes:
                # Take the last note block (most recent impression)
                blocks = notes.strip().split("\n\n## ")
                last = blocks[-1] if blocks else ""
                if last and not last.startswith("## "):
                    last = "## " + last
                entry += f"\n{last}"

            entries.append(entry)

        return "<PEOPLE>\n" + "\n\n".join(entries) + "\n</PEOPLE>"
