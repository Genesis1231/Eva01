"""
EVA's heartbeat — her autonomic layer for maintenance and self-monitoring.
"""

import asyncio
from pathlib import Path
from urllib.parse import urlparse

from config import logger
from eva.database.db import SQLiteHandler

PROBE_HOST = "1.1.1.1"    # coarse "am I online" egress probe target
PROBE_TIMEOUT = 3         # seconds — TCP reachability probe deadline

# The Room's screenshot desk (frontend/public/tmp).
SHOTS_DIR = Path(__file__).resolve().parents[2] / "frontend" / "public" / "tmp"
SHOTS_KEEP = 200          # most-recent look-*.png screenshots to retain

 # LangGraph checkpointer store (under DATA_DIR/database/)
GRAPH_DB = "eva_graph.db" 


class Heart:

    def __init__(
        self,
        db: SQLiteHandler,
        interval: int,
        embedding_url: str = "",
    ):
        self.db = db
        self.interval = interval  # 0 = disabled
        self.embedding_url = embedding_url

    async def start(self) -> None:
        """Beat forever — run maintenance checks on each pulse."""
        if not self.interval:
            logger.debug("Heart: heartbeat disabled (interval=0)")
            return

        logger.debug(f"Heart: tending vitals every {self.interval}s")
        while True:
            await asyncio.sleep(self.interval)
            await self._maintain()

    async def _maintain(self) -> None:
        """Run the vitals probes, log a single status line, warn on any failure,
        then run housekeeping.

        A failing check never kills the beat. Periodic maintenance (the Room's
        screenshot sweep, and future jobs like memory consolidation) plugs in here.
        """
        storage, network, embedding = await asyncio.gather(
            self._check_storage(),
            self._check_network(),
            self._check_embedding(),
            return_exceptions=True,
        )

        vitals = {
            "db": self._mark(storage),
            "net": self._mark(network),
            "embed": self._mark(embedding),
        }
        line = " ".join(f"{k}={v}" for k, v in vitals.items())
        if "down" in vitals.values():
            logger.warning(f"Heart: vitals — {line}")
        else:
            logger.info(f"Heart: vitals — {line}")

        await self._log_checkpoints()
        await self._sweep_shots()

    async def _sweep_shots(self) -> None:
        """Trim the Room's screenshot desk to the most recent SHOTS_KEEP files.

        Runs the blocking file I/O off the event loop; a failure here warns but
        never disturbs the beat.
        """
        def _trim() -> int:
            if not SHOTS_DIR.exists():
                return 0
            shots = sorted(
                SHOTS_DIR.glob("look-*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            stale = shots[SHOTS_KEEP:]
            for shot in stale:
                shot.unlink(missing_ok=True)
            return len(stale)

        try:
            removed = await asyncio.to_thread(_trim)
            if removed:
                logger.debug(f"Heart: swept {removed} stale screenshot(s)")
        except Exception as e:
            logger.warning(f"Heart: screenshot sweep failed — {e}")

    async def _log_checkpoints(self) -> None:
        """Log how many LangGraph checkpoints this session holds, and in total.

        A checkpoint is roughly one reasoning super-step. The current session is the
        newest thread_id (they're time-stamped). Reads LangGraph's internal
        `checkpoints` table; any failure (missing DB, schema change) is swallowed so
        the beat is never disturbed.
        """
        try:
            total = await self.db.fetchone(
                "SELECT count(*) AS ckpts, count(DISTINCT thread_id) AS sessions "
                "FROM checkpoints",
                db_name=GRAPH_DB,
            )
            session = await self.db.fetchone(
                "SELECT count(*) AS ckpts FROM checkpoints "
                "WHERE thread_id = (SELECT max(thread_id) FROM checkpoints)",
                db_name=GRAPH_DB,
            )
        except Exception as e:
            logger.debug(f"Heart: checkpoint stats unavailable — {e}")
            return

        if not total or not session:
            return
        logger.info(
            f"Heart: {session['ckpts']} checkpoints in current session · "
            f"total {total['ckpts']} / {total['sessions']}"
        )

    @staticmethod
    def _mark(result) -> str:
        """Render a check outcome: True→ok, False/error→down, None→off (skipped)."""
        if result is None:
            return "off"
        if result is True:
            return "ok"
        return "down"

    async def _check_storage(self) -> bool:
        """The local database answers a trivial query."""
        return await self.db.fetchone("SELECT 1") is not None

    async def _check_network(self) -> bool:
        """Internet Check — can we reach the public internet?"""
        return await self._reachable(PROBE_HOST, 443)

    async def _check_embedding(self) -> bool | None:
        """The local embedding server is listening. Skipped if not configured."""
        if not self.embedding_url:
            return None
        parsed = urlparse(self.embedding_url)
        if not parsed.hostname:
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        
        return await self._reachable(parsed.hostname, port)

    @staticmethod
    async def _reachable(host: str, port: int, timeout: int = PROBE_TIMEOUT) -> bool:
        """Cheap TCP reachability probe — no payload, no token cost."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), 
                timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False
