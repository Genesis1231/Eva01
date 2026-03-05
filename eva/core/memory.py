"""
EVA's memory — orchestrates journal, knowledge, and relationship reflection.

The checkpointer acts as a write-ahead log. On shutdown (or crash recovery),
raw messages are distilled into journal entries and the checkpointer is cleared.

Context assembly:
    journal (recent entries) + distilled current session → system prompt
"""
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import logger
from eva.agent.schema import PeopleReflection
from eva.core.journal import JournalDB
from eva.utils.prompt import load_prompt


class MemoryDB:
    """EVA's long-term memory — orchestration layer over JournalDB + PeopleDB."""

    def __init__(self, utility_model: str = None, people_db=None):
        self._journal = JournalDB()
        self._journal_prompt = load_prompt("journal")
        self._relationships_prompt = load_prompt("relationships")
        self._pen = init_chat_model(utility_model) if utility_model else None
        self._people = people_db
        logger.debug(f"MemoryDB: ready (utility_model={'in hand' if self._pen else 'missing'}).")


    # ── Distillation ─────────────────────────────────────────

    @staticmethod
    def distill(messages: list) -> list:
        """Collapse completed feel/speak tool cycles into clean AIMessages.

        Only distills PREVIOUS turns (before the last HumanMessage).
        The current turn stays raw so the ReAct loop can continue.
        """
        # Find the last HumanMessage — everything after it is the current turn
        last_human_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break

        history = messages[:last_human_idx] if last_human_idx > 0 else []
        current_turn = messages[last_human_idx:] if last_human_idx >= 0 else messages[:]

        result = []
        i = 0

        while i < len(history):
            msg = history[i]

            if not isinstance(msg, AIMessage) or not getattr(msg, 'tool_calls', None):
                if isinstance(msg, AIMessage) and not msg.content and not getattr(msg, 'tool_calls', None):
                    i += 1
                    continue
                result.append(msg)
                i += 1
                continue

            tool_calls = msg.tool_calls
            tool_names = {tc['name'] for tc in tool_calls}
            distillable = tool_names <= {'feel', 'speak'}

            if not distillable:
                result.append(msg)
                i += 1
                continue

            call_ids = {tc['id'] for tc in tool_calls}
            tool_msg_count = 0
            j = i + 1
            while j < len(history) and isinstance(history[j], ToolMessage):
                if history[j].tool_call_id in call_ids:
                    tool_msg_count += 1
                j += 1

            if tool_msg_count < len(call_ids):
                result.append(msg)
                i += 1
                continue

            parts = []
            for tc in tool_calls:
                name = tc['name']
                args = tc['args']
                if name == 'feel':
                    feeling = args.get('feeling', '')
                    parts.append(f"[I felt {feeling}]")
                elif name == 'speak':
                    text = args.get('text', '')
                    parts.append(f'I said: "{text}"')

            result.append(AIMessage(content="\n\n".join(parts)))

            i = j
            if i < len(history) and isinstance(history[i], AIMessage) \
                and not history[i].content \
                and not getattr(history[i], 'tool_calls', None):
                i += 1

        result.extend(current_turn)
        return result

    # ── Context Assembly ─────────────────────────────────────

    def prepare_context(self, messages: list, limit: int = 3) -> tuple[list, str]:
        """Distill current session messages + build journal context.

        Returns (distilled_messages, journal_summary).
        """
        distilled = self.distill(messages)

        # Get recent journal entries, limit to 3
        entries = self._journal.get_recent(limit)
        journal_summary = "\n\n".join(entries) if entries else ""

        return distilled, journal_summary

    @staticmethod
    def _text_content(content) -> str:
        """Extract text from message content (str or list of content blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        return str(content)

    # ── Flush ────────────────────────────────────────────────

    async def flush(self, messages: list, session_id: str = None) -> bool:
        """
        Summarize a full session into a journal entry via the utility LLM.
        Called on shutdown/recovery to save the session to the journal.
        """
        if not messages:
            logger.debug("MemoryDB: nothing to flush.")
            return False

        # Distill entire session (treat all messages as history)
        distilled = self.distill(messages)

        # Build conversation text from distilled messages
        parts = []
        for msg in distilled:
            if isinstance(msg, HumanMessage):
                parts.append(self._text_content(msg.content))
            elif isinstance(msg, AIMessage) and msg.content:
                parts.append(self._text_content(msg.content))

        if not parts:
            logger.debug("MemoryDB: distilled to nothing, skipping flush.")
            return False

        conversation = "\n".join(parts)

        # Journal the session via utility LLM
        journal, _ = await asyncio.gather(
            self._reflect_messages(conversation),
            self._reflect_people(conversation)
        )

        self._journal.add(journal, session_id)
        logger.debug(f"MemoryDB: journaled session ({len(journal.split())} words).")
        return True

    async def _reflect_messages(self, conversation: str) -> str:
        """Reflect on a conversation and return a journal entry."""
        if not self._pen:
            return conversation

        prompt = self._journal_prompt.replace("{conversation}", conversation)
        try:
            response = await self._pen.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"MemoryDB: message reflection failed — {e}")
            return conversation

    # ── Relationship Extraction ─────────────────────────────────────

    async def _reflect_people(self, conversation: str) -> None:
        """Extract per-person impressions from a session and append to PeopleDB."""
        if not self._pen or not self._people:
            return

        all_people = self._people.get_all()
        if not all_people:
            return

        # Only send people actually mentioned in the conversation
        conv_lower = conversation.lower()
        mentioned = {
            pid: p for pid, p in all_people.items()
            if p["name"].lower() in conv_lower
            or p["name"].split()[0].lower() in conv_lower.split()
        }
        if not mentioned:
            return

        # Build people list for the prompt
        people_lines = []
        for pid, person in mentioned.items():
            rel = person.get("relationship") or "no relationship noted"
            people_lines.append(f"{pid}: {person['name']} ({rel})")

        prompt = self._relationships_prompt.replace(
            "{conversation}", conversation
        ).replace(
            "{people}", "\n".join(people_lines)
        )

        try:
            structured_pen = self._pen.with_structured_output(PeopleReflection)
            reflection = await structured_pen.ainvoke(prompt)
        except Exception as e:
            logger.error(f"MemoryDB: relationship extraction failed — {e}")
            return

        # Write validated impressions to PeopleDB
        for entry in reflection.impressions:
            if entry.person_id in mentioned:
                self._people.append_notes(entry.person_id, entry.impression)
