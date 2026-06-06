"""EVA's moment memory — perceptual episodic memory, keyed per modality.

A *moment* = what was seen + what was said + Eva's first impression of it. It is
stored whole but **cued per channel**: a face (`img_key`) OR a phrase (`txt_key`)
alone can light the whole moment back up.

The keys are precomputed vectors handed in by the novelty gate's encoder — MomentDB
never embeds; it stores each modality's key in its own pure VectorIndex and keeps the
moment's mutable state (note, activation) in the `moments` table. The LLM
impression/importance call lives one layer up (as MemoryDB sits over JournalDB).

  remember → recall → reinforce
  (forgetting is retrieval failure, not data loss: nothing is deleted — stale moments
   just stop surfacing as activation × recency sinks in the recall ranking)
"""

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from config import logger
from eva.database.db import SQLiteHandler
from eva.database.vector_index import VectorIndex

_RECENCY_HALF_LIFE_DAYS = 7.0   # a moment's pull halves after a week without recall
_RECENCY_FLOOR = 0.1            # ...but never to nothing — a vivid one-off persists
_REINFORCE_GAIN = 0.3          # recall bump: activation += (1-activation)*gain (→ 1.0)


@dataclass
class Moment:
    id: str
    note: str
    score: float        # relevance × recency × activation, at recall time
    activation: float
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MomentDB:
    """Lifelong perceptual episodic memory: remember → recall → reinforce.
    Nothing is deleted — forgetting lives in the recall ranking, not the data."""

    def __init__(self, db: SQLiteHandler):
        self._db = db
        self._img = VectorIndex(db, prefix="moment_img")
        self._txt = VectorIndex(db, prefix="moment_txt")
        self._initialized = False

    async def init_db(self) -> None:
        if self._initialized:
            return
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS moments (
                id               TEXT PRIMARY KEY,
                note             TEXT,
                created_at       TIMESTAMP,
                last_recalled_at TIMESTAMP,
                activation       REAL,
                related_id       TEXT
            )
        """)
        self._initialized = True  # the vec0 indexes are created lazily on first upsert

    # ── remember ─────────────────────────────────────────────
    async def remember(
        self, note: str, img_key, txt_key, importance: float, related_id: str | None = None
    ) -> str:
        """Store a new moment. `importance` (her felt weight) seeds activation. Either key may be
        None — a silent moment is an image alone (and vice versa); it's simply not cued on that
        channel."""
        moment_id = uuid.uuid4().hex[:12]
        now = _now()
        try:
            await self._db.execute(
                """INSERT INTO moments
                (id, note, created_at, last_recalled_at, activation, related_id)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (moment_id, note, now, now, float(importance), related_id),
            )
            if img_key is not None:
                await self._img.upsert(moment_id, img_key)
            if txt_key is not None:
                await self._txt.upsert(moment_id, txt_key)
            return moment_id
        except Exception as e:
            logger.error(f"MomentDB: failed to remember — {e}")
            return ""

    # ── recall ───────────────────────────────────────────────
    async def recall(self, img_key=None, txt_key=None, limit: int = 5, min_score: float = 0.0) -> list[Moment]:
        """Recall moments cued by a face and/or a phrase. Either cue alone can hit
        (max over the available channels); ranked by relevance × recency × activation —
        the ranking IS the forgetting: faded moments score too low to surface."""
        relevance: dict[str, float] = {}
        for key, index in ((img_key, self._img), (txt_key, self._txt)):
            if key is None:
                continue
            for mid, cos in await index.search(key, limit=limit * 4, min_score=min_score):
                relevance[mid] = max(relevance.get(mid, -1.0), cos)   # max-fuse the cues
        if not relevance:
            return []

        placeholders = ",".join("?" * len(relevance))
        rows = list(await self._db.fetchall(
            f"""SELECT id, note, activation, last_recalled_at, created_at FROM moments
            WHERE id IN ({placeholders})""",
            tuple(relevance),
        ))
        now = datetime.now(timezone.utc)
        moments = [
            Moment(
                r["id"], r["note"],
                relevance[r["id"]] * self._recency(r["last_recalled_at"], now) * r["activation"],
                r["activation"], r["created_at"],
            )
            for r in rows
        ]
        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:limit]

    # ── reinforce ────────────────────────────────────────────
    async def reinforce(self, moment_id: str) -> None:
        """A recall strengthens the trace (diminishing toward 1.0) and resets its clock."""
        await self._db.execute(
            """UPDATE moments
            SET activation = MIN(1.0, activation + (1.0 - activation) * ?), last_recalled_at = ?
            WHERE id = ?""",
            (_REINFORCE_GAIN, _now(), moment_id),
        )

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _recency(last_recalled_at: str, now: datetime) -> float:
        try:
            age_days = max((now - datetime.fromisoformat(last_recalled_at)).total_seconds() / 86400, 0.0)
        except Exception:
            return 1.0
        return max(_RECENCY_FLOOR, math.exp(-math.log(2) * age_days / _RECENCY_HALF_LIFE_DAYS))
