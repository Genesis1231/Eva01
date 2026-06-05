"""
eva/utils/feed.py — Eva's outward feed: she posts text to the Room.
"""

import asyncio
import httpx

from config import logger, eva_configuration
from eva.subconscious.mood import surfaced_emotions

_feed_client: httpx.AsyncClient | None = None
_feed_tasks: set[asyncio.Task] = set()


def mood_labels(mood: list[float] | None, top_k: int = 3) -> str:
    """The Room's view of the mood vector"""
    return ", ".join(label for label, _ in surfaced_emotions(mood, top_k))


async def _post(kind: str, text: str) -> None:
    """Do the POST. Never raises."""
    global _feed_client
    if _feed_client is None:
        _feed_client = httpx.AsyncClient(timeout=3)  # reused; short timeout
    try:
        await _feed_client.post(eva_configuration.FEED_URL, json={"kind": kind, "text": text})
    except Exception as e:
        logger.warning(f"Feed_post: dropped {kind} — {e}")


def feed_post(kind: str, text) -> None:
    """Fire-and-forget post to the Room"""
    
    if not eva_configuration.FEED_URL:
        return
    
    if kind == "mood":
        text = mood_labels(text)
    
    task = asyncio.create_task(_post(kind, text))
    _feed_tasks.add(task)
    task.add_done_callback(_feed_tasks.discard)
