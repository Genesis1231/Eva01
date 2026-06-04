"""
EVA's heartbeat — her autonomic layer for maintenance and self-monitoring.
"""

import asyncio
from urllib.parse import urlparse

from config import logger
from eva.database.db import SQLiteHandler

PUBLIC_HOST = "1.1.1.1"   # coarse "am I online" egress probe target
PROBE_TIMEOUT = 3         # seconds — TCP reachability probe deadline


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
        """Run the vitals probes, log a single status line, warn on any failure.

        A failing check never kills the beat. Future maintenance (e.g.
        MomentDB.forget() once the moment store is wired) plugs in here.
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
            logger.debug(f"Heart: vitals — {line}")

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
        return await self._reachable(PUBLIC_HOST, 443)

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
