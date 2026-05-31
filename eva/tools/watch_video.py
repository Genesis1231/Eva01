"""
YouTube utilities — search, metadata, transcript extraction, video analysis.
Used by search.py and read.py tools.
"""

import asyncio
import html
import re
import urllib.request
from typing import Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import yt_dlp

from config import logger, eva_configuration as config
from eva.tools import ToolError
from eva.utils.prompt import load_prompt
from eva.tools._shared.video_analyzer import VideoAnalyzer
from eva.utils.feed import feed_post

_MAX_VIDEO_DURATION = 900  # 15 minutes — beyond this, use transcript
_analyzer: VideoAnalyzer | None = None
_summarizer = None


def _get_analyzer() -> VideoAnalyzer | None:
    global _analyzer
    if _analyzer is None:
        _analyzer = VideoAnalyzer()
    return _analyzer


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = init_chat_model(config.UTILITY_MODEL, max_tokens=150)
    return _summarizer

def _fetch_metadata(url: str) -> Dict[str, Any]:
    """Fetch video metadata (duration, subs, etc.) without downloading."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False) #type: ignore


def _extract_transcript(info: Dict[str, Any]) -> str | None:
    """Download and clean VTT transcript. Returns plain text or None."""
    subs = info.get("subtitles", {})
    auto = info.get("automatic_captions", {})
    en_subs = subs.get("en") or auto.get("en")
    if not en_subs:
        return None

    vtt_url = next((s["url"] for s in en_subs if s["ext"] == "vtt"), None)
    if not vtt_url:
        return None

    try:
        raw = urllib.request.urlopen(vtt_url, timeout=10).read().decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to download transcript: {e}")
        return None

    lines: list[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if "-->" in line or re.match(r"^\d+$", line):
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and (not lines or clean != lines[-1]):
            lines.append(clean)

    if not lines:
        return None

    return html.unescape(" ".join(lines))


async def _summarize_transcript(title: str, description: str, transcript: str) -> str:
    """Compress a long transcript into ~100 words via the utility model."""
    prompt = load_prompt("describe_transcript").format(
        title=title, description=description, transcript=transcript
    )
    try:
        response = await _get_summarizer().ainvoke(prompt)
        return str(response.content)
    except Exception as e:
        logger.error(f"Transcript summarization failed: {e}")
        return description[:500] + "..."


@tool
async def watch_video(url: str) -> str:
    """Watch a video by URL. I use this when I want to know what's in a video."""
    if not url:
        return "I need a URL to watch."

    feed_post("video", url)  # the clip on her desk → the Room canvas, while she watches

    is_youtube = "youtube.com" in url or "youtu.be" in url

    # YouTube: check duration, use transcript for long videos
    if is_youtube:
        try:
            info = await asyncio.to_thread(_fetch_metadata, url)
        except Exception as e:
            logger.error(f"Failed to fetch video metadata: {e}")
            raise ToolError(str(e), tool_name="watch_video") from e

        duration = info.get("duration") or 0
        title = info.get("title", "Untitled")

        if duration <= _MAX_VIDEO_DURATION:
            # Short video — analyze with Gemini
            return await _analyze_video(url)

        # Long video — use transcript, summarize before returning
        transcript = await asyncio.to_thread(_extract_transcript, info)
        description = info.get("description") or ""
        if transcript:
            summary = await _summarize_transcript(title, description, transcript)
            return f"I read the transcript of '{title}':\n\n{summary}"

        return f"This video is too long to watch. Description: {description[500:] if description else 'No description'}..."

    # Non-YouTube: just analyze directly
    return await _analyze_video(url)


async def _analyze_video(url: str) -> str:
    """Send video to Gemini for analysis."""
    try:
        analyzer = _get_analyzer()
        if analyzer is None:
            return "Video Analyzer is not available. Cannot watch video."

        summary, error = await analyzer.analyze(url)
    except Exception as e:
        logger.error(f"Video analysis failed for {url} — {e}")
        raise ToolError(str(e), tool_name="watch_video") from e

    if not summary:
        return f"I couldn't watch the video at {url}. {error}"

    return f"I just watched {url}\n\n{summary}"
