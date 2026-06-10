"""Reusable semantic vector store backed by SQLite + sqlite-vec (vec0).

A *pure* vector store: callers hand in precomputed vectors and it does exact KNN
over the whole store via sqlite-vec's `vec0` virtual table (cosine). 
"""

import re

import numpy as np
import sqlite_vec

from config import logger
from eva.database.db import SQLiteHandler


def _blob(vector) -> bytes:
    """Pack a list/array of floats into sqlite-vec's float32 wire format."""
    return sqlite_vec.serialize_float32(np.asarray(vector, dtype=np.float32).tolist())


class VectorIndex:
    """Save and search vectors via sqlite-vec KNN. Embedding + ranking live in the caller."""

    def __init__(self, db: SQLiteHandler, prefix: str):
        self._db = db
        self._table = f"{prefix}_vec"
        self._ready = False
        self._dim: int | None = None

    async def _ensure(self, dim: int) -> None:
        """Create the vec0 table once, at the dimension of the first vector seen. On a later
        run the table already exists — _exists() recovers its dimension, so the check below
        compares against the real table, not this process's first vector."""
        if not self._ready:
            await self._exists()
        if self._ready:
            if dim != self._dim:
                raise ValueError(f"dimension mismatch: table has {self._dim}d, got {dim}d")
            return
        await self._db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} "
            f"USING vec0(entry_id TEXT PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
        )
        self._dim = dim
        self._ready = True

    async def _exists(self) -> bool:
        """Check if the vec0 table exists; if so, recover its dimension from the stored DDL."""
        if self._ready:
            return True
        row = await self._db.fetchone(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (self._table,)
        )
        match = re.search(r"float\[(\d+)\]", row["sql"]) if row else None
        if match:
            self._dim = int(match.group(1))
            self._ready = True
        return self._ready

    async def upsert(self, entry_id: str, vector) -> None:
        """Store a precomputed vector (replacing any prior one for entry_id)."""
        
        try:
            await self._ensure(len(vector))
            blob = _blob(vector)
            # vec0 has no native upsert; delete+insert must be one transaction or a
            # crash between them silently drops the entry.
            await self._db.execute_transaction([
                (f"DELETE FROM {self._table} WHERE entry_id = ?", (entry_id,)),
                (f"INSERT INTO {self._table}(entry_id, embedding) VALUES (?, ?)", (entry_id, blob)),
            ])
        except Exception as e:
            logger.error(f"VectorIndex({self._table}): upsert failed — {e}")

    async def search(
        self, 
        query_vector, 
        limit: int = 5, 
        min_score: float = 0.0
    ) -> list[tuple[str, float]]:
        """Exact cosine KNN over the whole store. Returns (entry_id, cosine) sorted by
        similarity. Raw similarity only — the caller applies any recency/importance weighting."""
        
        if not await self._exists():
            return []
        try:
            rows = list(await self._db.fetchall(
                f"""SELECT entry_id, distance FROM {self._table}
                WHERE embedding MATCH ? AND k = ?""",
                (_blob(query_vector), limit),
            ))
        except Exception as e:
            logger.error(f"VectorIndex({self._table}): search failed — {e}")
            return []
        # sqlite-vec cosine distance = 1 - cosine_similarity
        return [(r["entry_id"], 1.0 - r["distance"]) for r in rows if (1.0 - r["distance"]) >= min_score]

    async def delete(self, entry_id: str) -> None:
        try:
            await self._db.execute(f"DELETE FROM {self._table} WHERE entry_id = ?", (entry_id,))
        except Exception as e:
            logger.error(f"VectorIndex({self._table}): delete failed — {e}")
