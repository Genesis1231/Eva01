"""
Screen: handles visual actions (watch, etc.) from ActionBuffer.
Logs for now — frontend WebSocket wiring comes later.
"""

from config import logger
from .action_buffer import ActionBuffer, ActionEvent


class Screen:
    """Visual action handler — EVA's screen output."""

    def register(self, buffer: ActionBuffer) -> None:
        """Register visual action handlers on the action buffer."""
        buffer.on("watch", self._handle_watch)

    async def _handle_watch(self, event: ActionEvent) -> None:
        """Handle watch: queue a YouTube video for display."""
        meta = event.metadata or {}
        title = meta.get("title", "Unknown")
        channel = meta.get("channel", "Unknown")
        logger.info(f"Screen: watch '{title}' by {channel} — video_id={event.content}")
