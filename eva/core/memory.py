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
    def distill(messages: list, full: bool = False, journal: bool = False) -> list:
        """Collapse completed feel/speak tool cycles into clean AIMessages.
        to de-noise and preserve the essence of feelings and speech for memory/journal.
        SAVE TOKENS too!

        By default, only distills PREVIOUS turns (before the last HumanMessage).
        The current turn stays raw so the ReAct loop can continue.
        When full=True, distills ALL messages (used by flush at shutdown).
        When journal=True, truncates tool results to their first line — keeps the
        action reference ("I read ...", "I found ...") but drops bulky content
        so the journal captures experience, not regurgitated external data.
        """
        if full:
            history = messages
            current_turn = []
        else:
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
            has_speak_call = any(tc.get("name") == "speak" for tc in tool_calls)
            call_ids = {tc['id'] for tc in tool_calls}

            # Collect matching ToolMessages
            tool_msgs = {}
            j = i + 1
            while j < len(history) and isinstance(history[j], ToolMessage):
                if history[j].tool_call_id in call_ids:
                    tool_msgs[history[j].tool_call_id] = history[j].content
                j += 1

            # Only distill if all tool results are present
            if len(tool_msgs) < len(call_ids):
                result.append(msg)
                i += 1
                continue

            # Use tool return values as the distilled content.
            # In journal mode, keep only the first line — external content
            # (webpage summaries, search results) lives after the first \n.
            parts = []
            for tc in tool_calls:
                content = tool_msgs.get(tc['id'])
                if not content:
                    continue
                if journal:
                    content = content.split('\n', 1)[0]
                if tc.get('name') == 'speak':
                    content = f'I said: "{content}"'
                parts.append(content)
            if parts:
                result.append(AIMessage(content="\n\n".join(parts)))

            i = j
            # ToolNode often returns an extra AI echo after `speak`; tool output already
            # captures the spoken line, so skip the immediate non-tool AI follow-up.
            if i < len(history) and isinstance(history[i], AIMessage) \
                and not getattr(history[i], 'tool_calls', None):
                if has_speak_call or not history[i].content:
                    i += 1

        # Strip metadata bloat (API signatures, usage stats) from current turn
        for msg in current_turn:
            if isinstance(msg, AIMessage):
                result.append(AIMessage(
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                    id=msg.id,
                ))
            else:
                result.append(msg)
        return result

    # ── Context Assembly ─────────────────────────────────────

    async def prepare_context(self, messages: list, limit: int = 5) -> tuple[list, str]:
        """Distill current session messages + build journal context.

        Returns (distilled_messages, journal_summary).
        """
        distilled = self.distill(messages)

        # Get recent journal entries, limit to 5
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

    async def flush(self, messages: list, session_id: str) -> None:
        """
        Summarize a full session into a journal entry via the utility LLM.
        Called on shutdown/recovery to save the session to the journal.
        """
        if not messages:
            logger.debug("MemoryDB: nothing to flush.")
            return

        # Distill entire session — full=True treats all messages as history,
        # journal=True truncates tool results to first line (experience > content)
        distilled = self.distill(messages, full=True, journal=True)

        # Build conversation text from distilled messages
        parts = []
        for msg in distilled:
            parts.append(self._text_content(msg.content))

        if not parts:
            logger.debug("MemoryDB: distilled to nothing, skipping flush.")
            return

        conversation = "\n\n".join(parts)
        logger.debug(f"MemoryDB: writing journal entry:\n{conversation}")
        
        # Skip journaling for trivially short sessions (e.g. a single observation)
        if len(conversation.split()) < 30:
            logger.debug("MemoryDB: session too short to journal, skipping.")
            return
        
        try:
            # Journal the session via utility LLM
            journal, _ = await asyncio.gather(
                self._reflect_messages(conversation),
                self._reflect_people(conversation)
            )

            await self._journal.add(journal, session_id)
            logger.debug(f"MemoryDB: journaled session: {journal}.")
            return
        except Exception as e:
            logger.error(f"MemoryDB: failed to flush memory — {e}")
            return

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