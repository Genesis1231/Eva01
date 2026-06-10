"""EVA's look_at tool — visual perception of webpages and documents."""

import asyncio
import io
import shutil
import time
from pathlib import Path

from html2image import Html2Image
from langchain_core.tools import tool
from PIL import Image

from config import logger
from eva.tools import ToolError
from eva.utils.format import image_uri
from eva.utils.feed import feed_post

_VIEWPORT = (1280, 800)
_VISION_SIZE = (800, 500)

# Where screenshots land so the Room can show them. frontend/public/ is served by
# Vite at the web root, so a file here is reachable at /tmp/<name>. 
_TMP_DIR = Path(__file__).resolve().parents[2] / "frontend" / "public" / "tmp"


def _pick_browser() -> str:
    for name in ("google-chrome", "google-chrome-stable",
                 "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("No Chrome/Chromium binary found in PATH.")

def _screenshot(url: str) -> tuple[bytes, str] | None:
    """Take a viewport screenshot, save it under a unique name in the Room's served folder,
    return (resized JPEG bytes, served URL)."""

    logger.debug(f"Taking screenshot of {url}...")
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        hti = Html2Image(
            browser_executable=_pick_browser(),
            output_path=str(_TMP_DIR),
            size=_VIEWPORT,
            custom_flags=[
                "--headless=new",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--hide-scrollbars",
                "--log-level=3",
                # Look like a real browser
                "--disable-blink-features=AutomationControlled",
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36",
            ],
        )
    except Exception as e:
        logger.error(f"Failed to initialize Html2Image: {e}")
        return None

    name = f"look-{time.time_ns()}.png"
    hti.screenshot(url=url, save_as=name)
    png_path = _TMP_DIR / name
    if not png_path.exists():
        logger.error("Screenshot file not found — failed to look at image.")
        return None

    img = Image.open(png_path)
    img = img.resize(_VISION_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), f"/tmp/{name}"


@tool
async def look_at(url: str) -> str | list:
    """I look at a webpage or document quickly before I decide to read or act on it."""
    try:
        shot = await asyncio.to_thread(_screenshot, url)
        if shot is None:
            return "Screenshot failed — I can't see."
        jpeg, shot_url = shot

        # Put what I'm looking at on my desk in the Room — the unique name makes it repaint.
        feed_post("image", shot_url)

        # Hand the screenshot straight to the main model in tool message.
        return [
            {"type": "text", "text": f"This is what I see at {url}:"},
            {"type": "image_url", "image_url": {"url": image_uri(jpeg, "image/jpeg")}},
        ]
    except Exception as e:
        logger.error(f"Tools: look_at tool error: {e}")
        raise ToolError(str(e), tool_name="look_at") from e
