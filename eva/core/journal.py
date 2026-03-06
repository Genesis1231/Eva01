"""
EVA's journal — episodic memory stored in SQLite.

Pure database operations: write entries, read recent, create tables.
Orchestration (flush, distill, LLM calls) lives in memory.py.
"""

import uuid
from datetime import datetime, timezone
from typing import List

from config import logger, DATA_DIR
from eva.core.db import SQLiteHandler


class JournalDB:
    """EVA's journal — episodic memory store."""
    
    def __init__(self, db: SQLiteHandler):
        self._db = db
        self._initialized = False

    async def init_db(self) -> None:
        """initialize_database."""
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
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                created_at  TIMESTAMP
            )
            """,
        )
        self._initialized = True

    async def add(self, content: str, session_id: str) -> str:
        """Write an episode to the journal. Returns the entry id."""
        
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

    async def get_recent(self, limit: int = 1) -> List[str]:
        """Get recent journal entries — today's entries, or last session's if none today."""
        
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        rows = list(await self._db.fetchall(
            "SELECT content FROM journal WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (today_start, limit),
        ))

        if rows:
            return [r["content"] for r in reversed(rows)]

        # Nothing today — grab last session's entries
        rows = list(await self._db.fetchall(
            "SELECT content FROM journal ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ))
        return [r["content"] for r in reversed(rows)]
