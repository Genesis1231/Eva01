"""
EVA's memory — orchestrates journal, knowledge, and relationship reflection.

The checkpointer acts as a write-ahead log. On shutdown (or crash recovery),
raw messages are distilled into journal entries and the checkpointer is cleared.

Context assembly:
    journal (recent entries) + distilled current session → system prompt
"""
import asyncio
from typing import Iterable, cast
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import logger
from eva.agent.schema import PeopleReflection
from eva.core.journal import JournalDB
from eva.core.people import PeopleDB
from eva.utils.prompt import load_prompt


class MemoryDB:
    """EVA's long-term memory — orchestration layer over JournalDB + PeopleDB."""

    def __init__(
        self,
        utility_model: str,
        people_db: PeopleDB,
        journal_db: JournalDB,
    ):
        self._journal = journal_db
        self._people = people_db
        self._journal_prompt = load_prompt("journal")
        self._relationships_prompt = load_prompt("relationships")
        self._pen = init_chat_model(utility_model)

        # Person IDs seen during the current session, populated by graph orchestration.
        self._session_people_ids: set[str] = set()
        
        logger.debug(f"MemoryDB: ready (utility_model={utility_model}).")

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

    async def prepare_context(self, messages: list, limit: int = 3) -> tuple[list, str]:
        """Distill current session messages + build journal context.

        Returns (distilled_messages, journal_summary).
        """
        distilled = self.distill(messages)

        # Get recent journal entries, limit to 3
        entries = await self._journal.get_recent(limit)
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

    async def flush(self, messages: list, session_id: str) -> bool:
        """
        Summarize a full session into a journal entry via the utility LLM.
        Called on shutdown/recovery to save the session to the journal.
        """
        if not messages:
            logger.debug("MemoryDB: nothing to flush.")
            self.clear_session_people()
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
            self.clear_session_people()
            return False

        conversation = "\n".join(parts)
        logger.debug(f"MemoryDB: writing journal entry for conversation:\n{conversation}")
        
        try:
            # Journal the session via utility LLM
            journal, _ = await asyncio.gather(
                self._reflect_messages(conversation),
                self._reflect_people(conversation)
            )

            await self._journal.add(journal, session_id)
            logger.debug(f"MemoryDB: journaled session: {journal}.")
            return True
        finally:
            self.clear_session_people()

    async def _reflect_messages(self, conversation: str) -> str:
        """Reflect on a conversation and return a journal entry."""
        if not self._pen:
            return conversation

        prompt = self._journal_prompt.format(conversation=conversation)
        try:
            response = await self._pen.ainvoke(prompt)
            return self._text_content(response.content)

        except Exception as e:
            logger.error(f"MemoryDB: message reflection failed — {e}")
            return conversation

    # ── Relationship Extraction ─────────────────────────────────────
    def add_people_to_session(self, person_ids: Iterable[str]) -> None:
        """Track people seen this session so relationship reflection can use IDs, not names."""
        for person_id in person_ids:
            self._session_people_ids.add(person_id)

    def clear_session_people(self) -> None:
        """Reset tracked session people after flush completes."""
        self._session_people_ids.clear()
    
    async def _reflect_people(self, conversation: str) -> None:
        """Extract per-person impressions from a session and append to PeopleDB."""

        if not self._session_people_ids:
            logger.debug("MemoryDB: no people seen this session, skipping relationship reflection.")
            return

        await self._people.touch(self._session_people_ids)
        mentioned = self._people.get_many(self._session_people_ids)
        if not mentioned:
            logger.debug("MemoryDB: no people in the session for relationship reflection.")
            return
        
        prompt_people = self._people.render_people(mentioned)
        prompt = self._relationships_prompt.format(
            conversation=conversation, 
            people=prompt_people
        ) 

        try:
            structured_pen = self._pen.with_structured_output(PeopleReflection)
            reflection = cast(PeopleReflection, await structured_pen.ainvoke(prompt))
        except Exception as e:
            logger.error(f"MemoryDB: relationship extraction failed — {e}")
            return

        await self._people.append_reflection_notes(
            mentioned=self._session_people_ids,
            impressions=reflection.impressions,
        )