"""EVA's look_at tool — visual perception of webpages and images."""

import asyncio
import io
import shutil
import tempfile
from pathlib import Path

from html2image import Html2Image
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from PIL import Image

from config import logger, eva_configuration as config
from eva.tools import ToolError
from eva.utils.format import image_uri
from eva.utils.prompt import load_prompt

_vision = None
_VIEWPORT = (1280, 800)
_VISION_SIZE = (800, 500)


def _pick_browser() -> str:
    for name in ("google-chrome", "google-chrome-stable", 
                 "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("No Chrome/Chromium binary found in PATH.")

def _screenshot(url: str) -> bytes:
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
            return f"I can't look at the url: {e}"

        hti.screenshot(url=url, save_as="shot.png")
        png_path = Path(tmp) / "shot.png"
        if not png_path.exists():
            raise FileNotFoundError("Screenshot failed — no image produced.")

        img = Image.open(png_path)
        img = img.resize(_VISION_SIZE, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _get_vision():
    global _vision
    if _vision is None:
        _vision = init_chat_model(config.VISION_MODEL, temperature=0.1)
    return _vision


@tool
async def look_at(url: str) -> str:
    """I look at a webpage or image to see what's there before I decide to read or act on it."""
    try:
        jpeg = await asyncio.to_thread(_screenshot, url)

        logger.debug(f"Analyzing screenshot...")
        prompt = load_prompt("look_at").format(title="", url=url)
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_uri(jpeg, "image/jpeg")}},
        ])
        response = await _get_vision().ainvoke([message])
        description = str(response.content)

        return f"I looked at {url}:\n{description}\n"
    except Exception as e:
        logger.error(f"look_at error: {e}")
        raise ToolError(str(e), tool_name="look_at") from e
