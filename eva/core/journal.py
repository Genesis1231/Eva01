"""
EVA's journal — episodic memory stored in SQLite.

Pure database operations: write entries, read recent, search semantically.
Orchestration (flush, distill, LLM calls) lives in memory.py.
"""

import math
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from config import logger
from eva.database.db import SQLiteHandler
from eva.database.embeddings import EmbeddingEngine
from eva.database.vector_index import VectorIndex

# Gentle recency boost on recall — 1.0 today, easing toward a 0.7 floor.
_RECENCY_HALF_LIFE_DAYS = 14.0


class JournalDB:
    """EVA's journal — episodic memory store."""

    def __init__(
        self,
        db: SQLiteHandler,
        vectors: Optional[VectorIndex] = None,
        embedder: Optional[EmbeddingEngine] = None,
    ):
        self._db = db
        self._vectors = vectors
        self._embedder = embedder
        self._initialized = False

    @property
    def _semantic(self) -> bool:
        """Vector recall needs both an index to store in and a live embedder."""
        return (
            self._vectors is not None
            and self._embedder is not None
            and self._embedder.enabled
        )

    @staticmethod
    def _recency(created_at: str, now: datetime) -> float:
        try:
            age_days = max((now - datetime.fromisoformat(created_at)).total_seconds() / 86400, 0.0)
        except Exception:
            return 1.0
        return max(0.7, math.exp(-0.2 * age_days / _RECENCY_HALF_LIFE_DAYS))

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
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS journal_source (
                entry_id    TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                FOREIGN KEY(entry_id) REFERENCES journal(id) ON DELETE CASCADE
            )
            """,
        )
        self._initialized = True  # the vec0 index is created lazily on first upsert

    async def add(self, content: str, session_id: str, source: str = "") -> str:
        """Write an episode to the journal. Returns the entry id.

        Args:
            content: LLM-summarized journal entry (what EVA reads back).
            session_id: Session identifier for grouping entries.
            source: Raw conversation text. Embedded instead of content
                    for richer semantic search. Falls back to content if empty.
        """
        entry_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                "INSERT INTO journal (id, content, session_id, created_at) VALUES (?, ?, ?, ?)",
                (entry_id, content, session_id, now)
            )
            if source:
                await self._db.execute(
                    "INSERT INTO journal_source (entry_id, source) VALUES (?, ?)",
                    (entry_id, source),
                )
            # Embed the source (rich) when available, fall back to content (summary)
            if self._semantic:
                embed_text = source or content
                vector = await self._embedder.embed_one(embed_text)
                if vector:
                    await self._vectors.upsert(entry_id, vector)
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

    async def get_semantic_context(self, query: str, limit: int = 5, min_score: float = 0.5) -> str:
        """Return formatted journal snippets semantically close to `query`.

        Raw cosine gates relevance (min_score, tuned to the Qwen embedding scale);
        recency only *orders* the survivors, so a relevant-but-old memory still
        surfaces. Presented chronologically."""
        if not self._semantic:
            return ""

        query_vector = await self._embedder.embed_one(query)
        if not query_vector:
            return ""

        hits = dict(await self._vectors.search(query_vector, limit=limit * 4, min_score=min_score))
        if not hits:
            return ""

        placeholders = ",".join("?" * len(hits))
        rows = list(await self._db.fetchall(
            f"SELECT id, content, created_at FROM journal WHERE id IN ({placeholders})",
            tuple(hits),
        ))
        if not rows:
            return ""

        now = datetime.now(timezone.utc)
        rows.sort(key=lambda r: hits[r["id"]] * self._recency(r["created_at"], now), reverse=True)
        chosen = sorted(rows[:limit], key=lambda r: r["created_at"])  # relevance-picked, shown in order
        return "\n\n".join(self._format_row(r) for r in chosen)
