"""
EVA's memory — journal (episodic) and knowledge (semantic) stores.

The checkpointer acts as a write-ahead log. On shutdown (or crash recovery),
raw messages are distilled into journal entries and the checkpointer is cleared.

Context assembly:
    journal (recent entries) + distilled current session → system prompt
"""

import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import logger

from pathlib import Path

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "database" / "eva.db"


class MemoryDB:
    """EVA's long-term memory — journal and knowledge stores."""

    def __init__(self):
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        logger.debug("MemoryDB: ready.")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS journal (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    session_id  TEXT,
                    created_at  TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    created_at  TIMESTAMP
                )
            """)

    # ── Journal ──────────────────────────────────────────────

    def add_journal(self, content: str, session_id: str = None) -> str:
        """Write an episode to the journal. Returns the entry id."""
        entry_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO journal (id, content, session_id, created_at) VALUES (?, ?, ?, ?)",
                    (entry_id, content, session_id, now),
                )
            logger.debug(f"MemoryDB: journal entry {entry_id} written ({len(content)} chars).")
            return entry_id
        except sqlite3.Error as e:
            logger.error(f"MemoryDB: failed to write journal — {e}")
            return ""

    def get_recent_journal(self, limit: int = 10) -> List[str]:
        """Get recent journal entries — today's entries, or last session's if none today."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content FROM journal WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (today_start, limit),
            ).fetchall()

            if rows:
                return [r["content"] for r in reversed(rows)]

            # Nothing today — grab last session's entries
            rows = conn.execute(
                "SELECT content FROM journal ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [r["content"] for r in reversed(rows)]

    # ── Knowledge ────────────────────────────────────────────

    def add_knowledge(self, content: str) -> str:
        """Write a knowledge entry. Returns the entry id."""
        entry_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO knowledge (id, content, created_at) VALUES (?, ?, ?)",
                    (entry_id, content, now),
                )
            return entry_id
        except sqlite3.Error as e:
            logger.error(f"MemoryDB: failed to write knowledge — {e}")
            return ""

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
            if i < len(history) and isinstance(history[i], AIMessage) and not history[i].content and not getattr(history[i], 'tool_calls', None):
                i += 1

        result.extend(current_turn)
        return result

    # ── Context Assembly ─────────────────────────────────────

    def prepare_context(self, messages: list) -> tuple[list, str]:
        """Distill current session messages + build journal context.

        Returns (distilled_messages, journal_summary).
        """
        distilled = self.distill(messages)

        entries = self.get_recent_journal()
        if entries:
            journal_summary = "\n\n".join(entries)
        else:
            journal_summary = ""

        return distilled, journal_summary

    # ── Flush ────────────────────────────────────────────────

    def flush(self, messages: list, session_id: str = None) -> bool:
        """Distill a full session into a journal entry. Called on shutdown/recovery."""
        if not messages:
            logger.debug("MemoryDB: nothing to flush.")
            return False

        # Distill entire session (treat all messages as history)
        distilled = self.distill(messages)

        # Build a text summary from distilled messages
        parts = []
        for msg in distilled:
            if isinstance(msg, HumanMessage):
                parts.append(f"Sensed: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                parts.append(msg.content)

        if not parts:
            logger.debug("MemoryDB: distilled to nothing, skipping flush.")
            return False

        summary = "\n".join(parts)
        self.add_journal(summary, session_id=session_id)
        logger.debug(f"MemoryDB: flushed session → journal ({len(parts)} entries).")
        return True
