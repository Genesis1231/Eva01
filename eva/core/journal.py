"""
EVA's journal — autobiographical record stored in SQLite.

A plain text diary: write entries, read recent. No embedding, no semantic
search — associative recall lives in MomentDB. Orchestration (flush, distill,
LLM calls) lives in memory.py.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from config import logger
from eva.database.db import SQLiteHandler


class JournalDB:
    """EVA's journal — autobiographical text record."""

    def __init__(self, db: SQLiteHandler):
        self._db = db
        self._initialized = False

    @staticmethod
    def _format_row(row) -> str:
        try:
            dt = datetime.fromisoformat(row["created_at"])
            ts = dt.strftime("%B %d, at %I%p")
            ts = ts.replace(" at 0", " at ")
            return f"[{ts}]\n {row['content']}"
        except Exception:
            return row["content"]

    async def init_db(self) -> None:
        if self._initialized:
            return
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS journal (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                session_id  TEXT,
                created_at  TIMESTAMP
            )
            """,
        )
        self._initialized = True

    async def add(self, content: str, session_id: str) -> str:
        """Write an episode to the journal. Returns the entry id.

        Args:
            content: LLM-summarized journal entry (what EVA reads back).
            session_id: Session identifier for grouping entries.
        """
        entry_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                "INSERT INTO journal (id, content, session_id, created_at) VALUES (?, ?, ?, ?)",
                (entry_id, content, session_id, now)
            )
            return entry_id
        except Exception as e:
            logger.error(f"JournalDB: failed to write journal — {e}")
            return ""

    async def get_recent(self, limit: int = 3) -> List[str]:
        """Get today's journal entries."""

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        rows = list(await self._db.fetchall(
            "SELECT content, created_at FROM journal WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (today_start, limit),
        ))

        if rows:
            return [self._format_row(r) for r in reversed(rows)]
        else:
            return []
