"""EVA's look_at tool — visual perception of webpages and images."""

import asyncio
import io
import shutil
import tempfile
from pathlib import Path

from html2image import Html2Image
from langchain_core.tools import tool
from PIL import Image

from config import logger
from eva.tools import ToolError
from eva.utils.format import image_uri

_VIEWPORT = (1280, 800)
_VISION_SIZE = (800, 500)


def _pick_browser() -> str:
    for name in ("google-chrome", "google-chrome-stable", 
                 "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("No Chrome/Chromium binary found in PATH.")

def _screenshot(url: str) -> bytes | None:
    """Take a viewport screenshot, resize, return JPEG bytes."""

    logger.debug(f"Taking screenshot of {url}...")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            hti = Html2Image(
                browser_executable=_pick_browser(),
                output_path=tmp,
                size=_VIEWPORT,
                custom_flags=[
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--log-level=3",
                ],
            )
        except Exception as e:
            logger.error(f"Failed to initialize Html2Image: {e}")
            return None

        hti.screenshot(url=url, save_as="shot.png")
        png_path = Path(tmp) / "shot.png"
        if not png_path.exists():
            raise FileNotFoundError("Screenshot failed — no image produced.")

        img = Image.open(png_path)
        img = img.resize(_VISION_SIZE, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


@tool
async def look_at(url: str) -> str | list:
    """I look at a webpage or image to see what's there before I decide to read or act on it."""
    try:
        jpeg = await asyncio.to_thread(_screenshot, url)
        if jpeg is None:
            return "Screenshot failed — I can't see."

        # Hand the screenshot straight to my own eyes (the brain's vision) — no separate vision model.
        # Returning content blocks: the tool layer wraps them into a ToolMessage and fills the tool_call_id.
        return [
            {"type": "text", "text": f"This is what I see at {url}:"},
            {"type": "image_url", "image_url": {"url": image_uri(jpeg, "image/jpeg")}},
        ]
    except Exception as e:
        logger.error(f"Tools: look_at tool error: {e}")
        raise ToolError(str(e), tool_name="look_at") from e
