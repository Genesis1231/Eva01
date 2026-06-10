"""EVA's moment memory — perceptual episodic memory, a pure per-modality store.

A *moment* is the settled trace an *Experience* leaves: stored whole, **cued per channel** so any
one key relights the whole moment — a face (`img_key`), a phrase (`txt_key`), the screen her actions
left (`canvas_key`), or her impression (`note_key`). A reflection (her journal) is the same row with
no senses (`kind='reflection'`, perceptual keys None, the note carrying the text).

Why a *pure* store: keys are precomputed vectors handed in by the encoder; MomentDB never embeds and
never calls an LLM (that lives one layer up) — it stays a thin, testable storage seam.

Why nothing is deleted: forgetting is retrieval failure, not data loss — stale moments sink in the
ranking, never erased. MVP ranks recall by relevance alone; the recency × activation forgetting
curve is a later slice.
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
    score: float        # MVP: relevance (max-fused cosine). recency × activation fold in later.
    activation: float
    created_at: str



class MomentDB:
    """Pure episodic store: memorize → recall → reinforce. Embeddings come from the caller."""

    def __init__(self, db: SQLiteHandler):
        self._db = db
        self._img = VectorIndex(db, prefix="moment_img")
        self._txt = VectorIndex(db, prefix="moment_txt")
        self._canvas = VectorIndex(db, prefix="moment_canvas")
        self._note = VectorIndex(db, prefix="moment_note")
        self._initialized = False

    async def init_db(self) -> None:
        if self._initialized:
            return
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS moments (
                id               TEXT PRIMARY KEY,
                note             TEXT,
                kind             TEXT DEFAULT 'moment',
                created_at       TIMESTAMP,
                last_recalled_at TIMESTAMP,
                activation       REAL,
                related_id       TEXT
            )
        """)
        self._initialized = True  # the vec0 indexes are created lazily on first upsert

    # ── memorize ─────────────────────────────────────────────
    async def memorize(
        self,
        note: str,
        img_key,
        txt_key,
        importance: float,
        canvas_key=None,
        note_key=None,
        kind: str = "moment",
        related_id: str | None = None,
    ) -> str:
        """Store a new moment."""
        
        moment_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                """INSERT INTO moments
                (id, note, kind, created_at, last_recalled_at, activation, related_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (moment_id, note, kind, now, now, float(importance), related_id),
            )
            for key, index in (
                (img_key, self._img), (txt_key, self._txt),
                (canvas_key, self._canvas), (note_key, self._note),
            ):
                if key is not None:
                    await index.upsert(moment_id, key)
            return moment_id
        except Exception as e:
            logger.error(f"MomentDB: failed to memorize — {e}")
            return ""

    # ── recall ───────────────────────────────────────────────
    async def recall(
        self, img_key=None, txt_key=None, canvas_key=None, note_key=None,
        limit: int = 5, min_score: float = 0.0,
    ) -> list[Moment]:
        """Recall by any combination of cues, **max-fused** over the channels present: recall is OR
        — one strong cue should relight the whole moment, so averaging (which dilutes a single hit)
        is wrong. 
        
        MVP ranks by relevance alone (forgetting curve is a later slice). Reflections have
        no perceptual keys, so only the note cue surfaces them."""
        
        relevance: dict[str, float] = {}
        for key, index in (
            (img_key, self._img), (txt_key, self._txt),
            (canvas_key, self._canvas), (note_key, self._note),
        ):
            if key is None:
                continue
            for mid, cos in await index.search(key, limit=limit * 4, min_score=min_score):
                relevance[mid] = max(relevance.get(mid, -1.0), cos)   # max-fuse the cues
        if not relevance:
            return []

        placeholders = ",".join("?" * len(relevance))
        rows = list(await self._db.fetchall(
            f"""SELECT id, note, activation, created_at FROM moments
            WHERE id IN ({placeholders})""",
            tuple(relevance),
        ))
        moments = [
            Moment(r["id"], r["note"], relevance[r["id"]], r["activation"], r["created_at"])
            for r in rows
        ]
        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:limit]

    async def reinforce(self, moment_id: str) -> None:
        """A recall strengthens the trace (diminishing toward 1.0) and resets its clock.
        Defined for the forgetting slice — not yet called on recall (MVP ranks by relevance)."""
        await self._db.execute(
            """UPDATE moments
            SET activation = MIN(1.0, activation + (1.0 - activation) * ?), last_recalled_at = ?
            WHERE id = ?""",
            (_REINFORCE_GAIN, datetime.now(timezone.utc).isoformat(), moment_id),
        )


    async def recent(self, limit: int = 5) -> list[str]:
        """Today's reflections, newest-N, returned oldest-first and timestamp-formatted — the
        journal context injected into her prompt each session. Read-only: never reinforces (this
        runs every boot). This is the diary view of the unified store (the old JournalDB's role)."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        rows = list(await self._db.fetchall(
            "SELECT note, created_at FROM moments WHERE kind = 'reflection' AND created_at >= ? "
            "ORDER BY created_at DESC LIMIT ?",
            (today_start, limit),
        ))
        return [self._format_row(r) for r in reversed(rows)] if rows else []

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _format_row(row) -> str:
        """Render a reflection as '[June 06, at 3PM]\\n <note>' for the prompt."""
        try:
            dt = datetime.fromisoformat(row["created_at"])
            ts = dt.strftime("%B %d, at %I%p").replace(" at 0", " at ")
            return f"[{ts}]\n {row['note']}"
        except Exception:
            return row["note"]

    @staticmethod
    def _recency(last_recalled_at: str, now: datetime) -> float:
        """Dormant — the forgetting curve. Wired into recall ranking by the forgetting slice
        (alongside reinforce-on-recall); defined here so that slice is a localized change."""
        try:
            age_days = max((now - datetime.fromisoformat(last_recalled_at)).total_seconds() / 86400, 0.0)
        except Exception:
            return 1.0
        return max(_RECENCY_FLOOR, math.exp(-math.log(2) * age_days / _RECENCY_HALF_LIFE_DAYS))
