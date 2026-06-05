"""EVA's peak tool — a quick glance through my own eyes (the webcam)."""

import asyncio
from langchain_core.tools import tool

from config import logger
from eva.senses.vision.vision_sense import CameraSense
from eva.tools import ToolError

_camera: CameraSense | None = None


def init(camera: CameraSense) -> None:
    """Share the live CameraSense."""
    global _camera
    _camera = camera


@tool
async def peak() -> str:
    """I use this to see what's in front of my camera right now."""
    if _camera is None or not _camera.is_available:
        return "I can't see right now — my camera is not available."
    try:
        frame = await asyncio.to_thread(_camera.capture_photo)   # sync cv2, thread-safe (webcam lock)
        view = await _camera.describe(frame)
        return f"I see: {view}" if view else "The intepreter broke, I don't know what I see."
    except Exception as e:
        logger.error(f"peak error: {e}")
        raise ToolError(str(e), tool_name="peak_tool") from e
