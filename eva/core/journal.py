"""
EVA's journal — episodic memory stored in SQLite.

Pure database operations: write entries, read recent, create tables.
Orchestration (flush, distill, LLM calls) lives in memory.py.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List

from config import logger, DATA_DIR


class JournalDB:
    """EVA's journal — episodic memory store."""

    def __init__(self):
        self._init_db()

    def _init_db(self) -> None:
        (DATA_DIR / "database").mkdir(parents=True, exist_ok=True)
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

    def _connect(self) -> sqlite3.Connection:
        db_path = DATA_DIR / "database" / "eva.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, content: str, session_id: str = None) -> str:
        """Write an episode to the journal. Returns the entry id."""
        entry_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO journal (id, content, session_id, created_at) VALUES (?, ?, ?, ?)",
                    (entry_id, content, session_id, now),
                )
            return entry_id
        except sqlite3.Error as e:
            logger.error(f"JournalDB: failed to write journal — {e}")
            return ""

    def get_recent(self, limit: int = 10) -> List[str]:
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
            logger.error(f"JournalDB: failed to write knowledge — {e}")
            return ""
