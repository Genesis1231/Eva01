"""
PromptConstructor:
    Constructs and formats prompts for the chat agent.
    Assembles various components into structured prompts by combining:
    - Persona and instruction templates loaded from files
    - Conversation history
"""

from typing import Any
from config import logger
from eva.utils.prompt import load_prompt

class PromptConstructor:
    """
    PromptConstructor class: Loads persona and instruction templates from files.
    """

    def __init__(self, people: dict[str, Any] | None = None):
        self.soul: str = load_prompt("SOUL") # default persona prompt
        self.instructions: str = load_prompt("INSTRUCTIONS") # default instructions prompt
        self.people: dict[str, Any] | None = people

    def build_system(self, timestamp: str) -> str:
        """Build the stable system prompt: persona + instructions + current time.

        Identity only. Volatile per-turn context (memory, people, mood) is built by build_context
        and injected next to the current turn by the Cortex, not glued into identity here."""
        return (
            f"<PERSONA>{self.soul}</PERSONA>\n\n"
            f"<INSTRUCTIONS>\n"
            f"{self.instructions}\n"
            f"</INSTRUCTIONS>\n\n"
            f"<CURRENT_TIME>{timestamp}</CURRENT_TIME>\n\n"
        )

    def build_context(
        self,
        memory: str = "",
        present_people: set[str] = set(),
        mood_block: str = "",
    ) -> str:
        """Build the volatile per-turn context: what's recalled/known + who's present + mood.

        Returned as a plain string for the Cortex to carry in a transient turn-side message (not the
        system prompt): it changes every turn and should read as 'what's true right now'. Returns ''
        when there's nothing, so the caller can skip the message entirely."""
        people_block = self._build_people_block(present_people)

        parts = []
        if people_block or memory:
            block = "<MEMORY>"
            if people_block:
                block += f"\n{people_block}"
            if memory:
                block += f"\n{memory}"
            block += "\n</MEMORY>"
            parts.append(block)

        if mood_block:
            parts.append(mood_block)

        return "\n\n".join(parts)

    def _build_people_block(self, present_people: set[str] | None) -> str :
        """Build <PEOPLE> block from face IDs currently visible to EVA."""
        
        if not present_people or not self.people:
            return ""

        entries = []
        for person_id in present_people:
            person = self.people.get(person_id)
            if not person:
                continue

            name = person["name"]
            rel = person.get("relationship") or "unknown"
            notes = person.get("notes") or ""

            entry = f"{name} ({rel})"
            # if notes:
            #     # Take the last note block (most recent impression)
            #     blocks = notes.strip().split("\n\n## ")
            #     last = blocks[-1] if blocks else ""
            #     if last and not last.startswith("## "):
            #         last = "## " + last
            #     entry += f"\n{last}"

            entries.append(entry)

        return "<PEOPLE>\n" + "\n\n".join(entries) + "\n</PEOPLE>"
