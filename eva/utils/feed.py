"""
eva/utils/feed.py — Eva's outward feed: she posts text to the Room.

Eva is a pure producer. Each post is a (kind, text) pair — kind says which part of
the Room it belongs to (mood, a sense/speech line, an artifact on her desk). The
Room (a Vite relay → SSE) fans it out to the browser, which routes by kind. The
Room is an observer, never a dependency: with no url, or no one listening, a post
is a silent no-op — Eva never stalls or fails for it.

`feed_post` is fire-and-forget: it schedules the POST and returns immediately, so
a slow or absent Room can't block Eva's reasoning. For kind="mood", pass the raw
mood vector — the display labels are rendered here, and an unchanged mood is
skipped (so re-posting the same mood every reasoning step doesn't spam the Room).
"""

import asyncio

import httpx

from config import logger, eva_configuration
from eva.subconscious.mood import surfaced_emotions

_feed_client: httpx.AsyncClient | None = None
_tasks: set[asyncio.Task] = set()
_last_mood: str | None = None


def mood_labels(mood: list[float] | None, top_k: int = 2) -> str:
    """The Room's view of the mood vector: the top-k labels, strongest first —
    e.g. "curious, tender" (no neutral, no percentages)."""
    return ", ".join(label for label, _ in surfaced_emotions(mood, top_k))


async def _post(kind: str, text: str) -> None:
    """Do the POST. Never raises."""
    global _feed_client
    if _feed_client is None:
        _feed_client = httpx.AsyncClient(timeout=3)  # reused; short timeout
    try:
        await _feed_client.post(eva_configuration.FEED_URL, json={"kind": kind, "text": text})
    except Exception as e:
        logger.debug(f"feed_post: dropped {kind} — {e}")


def feed_post(kind: str, text) -> None:
    """Fire-and-forget post to the Room — never blocks the caller, never raises.
    No-op when the feed is disabled (empty FEED_URL). For kind="mood", `text` is
    the raw mood vector: rendered to labels here, and skipped if unchanged.
    Call from within the running event loop (Eva's spine)."""
    if not eva_configuration.FEED_URL:
        return
    if kind == "mood":
        text = mood_labels(text)
        global _last_mood
        if text == _last_mood:
            return  # mood unchanged since last post — don't re-post every think
        _last_mood = text
    task = asyncio.create_task(_post(kind, text))
    _tasks.add(task)  # hold a ref so the task can't be GC'd mid-flight
    task.add_done_callback(_tasks.discard)
