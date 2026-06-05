"""
Screen: handles visual actions (watch, etc.) from ActionBuffer.
"""

import webbrowser

from config import logger
from eva.actions.action_buffer import ActionBuffer, ActionEvent
from eva.actions.base import BaseAction

class Browser(BaseAction):
    """Visual action handler — EVA's browser output."""

    def register(self, buffer: ActionBuffer) -> None:
        buffer.on("show", self._handle_show)

    async def _handle_show(self, event: ActionEvent) -> None:
        """Open a URL in the host browser."""
        url = event.content
        
        if not url or not url.startswith(("http://", "https://")):
            logger.error(f"Browser: invalid URL '{url}'")
            return
        
        logger.debug(f"Browser: opening {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.error(f"Browser: failed to open URL — {e}")
 
