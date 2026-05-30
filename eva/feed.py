"""
eva/feed.py — Eva's outward feed: she posts text to the Room, one line at a time.

Eva is a pure producer. Each post is a (kind, text) pair — kind says which part of
the Room it belongs to (her mood, a stream line, an artifact on her desk), text is
the content. Everything Eva emits is text. The Room (a Vite relay → SSE) fans it
out to the browser, which routes by kind. The Room is an observer, never a
dependency: with no url, or no one listening, post is a silent no-op — Eva never
stalls or fails for it.
"""

from __future__ import annotations

import httpx

from config import logger


class Feeder:
    """Posts (kind, text) events to the Room. Empty url ⇒ disabled (no-op)."""

    def __init__(self, url: str = "", timeout: float = 1.0) -> None:
        self._url = url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def post(self, kind: str, text: str) -> None:
        """Send one event to the Room. No-op if disabled; never raises."""
        if not self._url:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        try:
            await self._client.post(self._url, json={"kind": kind, "text": text})
        except Exception as e:
            logger.debug(f"Feeder: dropped {kind} — {e}")

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
