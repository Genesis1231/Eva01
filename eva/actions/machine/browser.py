"""
Screen: handles visual actions (watch, etc.) from ActionBuffer.
"""

import platform
import subprocess
import webbrowser

from config import logger
from ..action_buffer import ActionBuffer, ActionEvent
from ..base import BaseAction

_IS_WSL = "microsoft" in platform.release().lower()


def _open_url(url: str) -> None:
    """Open a URL in the host browser. Uses Windows browser on WSL2."""
    if _IS_WSL:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        webbrowser.open(url)

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
            _open_url(url)
        except Exception as e:
            logger.error(f"Browser: failed to open URL — {e}")
 
