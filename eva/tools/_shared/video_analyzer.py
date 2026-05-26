"""
VideoAnalyzer — processes and describes videos using vision models.
Handles YouTube URLs and local files, auto-optimizes for cost efficiency. 
"""

import asyncio
import base64
import os
from typing import Any, Dict, List, Tuple
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote

import aiofiles
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from config import logger
from eva.utils.prompt import load_prompt


class VideoAnalyzer:
    """
    Analyzes video content via a vision LLM.
    Supports YouTube URLs (streamed directly) and local files (optimized + base64).
    """

    def __init__(
        self, 
        model_name: str = "google_genai:gemini-2.5-flash", 
        temperature: float = 0.1
    ):
        self.model = init_chat_model(model_name, temperature=temperature)
        logger.debug(f"VideoAnalyzer: {model_name} ready.")

    def _optimize_video_sync(self, input_path: str, max_dimension: int = 480) -> str | None:
        """Ultra-compress video for cheap analysis. Runs synchronously."""
        output_path = f"optimized_{Path(input_path).stem}.mp4"

        try:
            duration_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_path,
            ]
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
            logger.debug(f"Video duration: {duration}s")

            optimized_fps = 1 if duration < 60 else 0.5
            scale_filter = f"scale='min({max_dimension},iw)':'min({max_dimension},ih)':force_original_aspect_ratio=decrease"

            ffmpeg_cmd = [
                'ffmpeg', '-i', input_path, '-y',
                '-vf', f'{scale_filter},fps={optimized_fps}',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '35',
                '-an', '-movflags', '+faststart',
                output_path,
            ]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)

            original_tokens = int(duration * 258)
            optimized_tokens = int(duration * 66 * 0.5)
            if original_tokens > 0:
                savings = ((original_tokens - optimized_tokens) / original_tokens) * 100
                logger.debug(f"Video optimization savings: {savings:.1f}%")

            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Video optimization FFmpeg failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error optimizing video: {e}")
            return None

    async def _optimize_video(self, input_path: str) -> str | None:
        return await asyncio.to_thread(self._optimize_video_sync, input_path)

    async def convert_file(self, video_path: str) -> str | None:
        """Optimize and base64-encode a local video file."""
        optimized_video = await self._optimize_video(video_path)
        if not optimized_video:
            return None

        try:
            async with aiofiles.open(optimized_video, "rb") as f:
                encoded_video = base64.b64encode(await f.read()).decode("utf-8")
            return encoded_video
        except Exception as e:
            logger.error(f"Error encoding video {video_path}: {e}")
            return None
        finally:
            if optimized_video and os.path.exists(optimized_video):
                os.remove(optimized_video)

    async def analyze(self, url: str, context: str = "") -> Tuple[str | None, str]:
        """
        Analyze a video and return a text description and error message.
        Supports YouTube URLs (streamed via Gemini) and local files.
        """
        if not url or not isinstance(url, str):
            logger.error("No video URL provided.")
            return None, "No video URL provided."

        prompt = load_prompt("describe_video").format(context=context)
        content = [{"type": "text", "text": prompt}]

        if url.startswith(("http://", "https://")):
            if "youtube.com" in url or "youtu.be" in url:
                content.append({
                    "type": "media", 
                    "mime_type": "video/mp4", 
                    "file_uri": url
                })
            else:
                logger.error(f"Unsupported video URL: {url}")
                return None, f"Unsupported video URL: {url}"
        else:
            extension = extract_video_extension(url)
            if extension not in ("mp4", "mov", "avi", "mkv", "wmv", "mpg", "mpeg", "3gpp"):
                logger.error(f"Unsupported video format: {extension}")
                return None, f"Unsupported video format: {extension}"

            encoded_video = await self.convert_file(url)
            if not encoded_video:
                return None, "Failed to encode video."

            content.append({
                "type": "media",
                "data": encoded_video,
                "mime_type": f"video/{extension}",
            })

        try:
            logger.debug(f"Analyzing video: {url}")
            response = await self.model.ainvoke([HumanMessage(content=content)]) #type: ignore
            return response.content, ""
        except Exception as e:
            logger.error(f"Failed to analyze video: {e}")
            return None, f"{e}"


def extract_video_extension(url: str) -> str:
    """Extract file extension from a URL or path, handling edge cases."""
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    last = path.rsplit("/", 1)[-1]

    if ";" in last:
        last = last.split(";", 1)[0]
    if not last or last.endswith("/"):
        return ""
    if "." not in last or last.startswith("."):
        return ""

    return last.rsplit(".", 1)[-1].lower()
