"""EVA's search tool — finds information across the web."""

import re
import asyncio
from typing import Any
from langchain_core.tools import tool
from config import logger
from typing import Literal
from eva.tools import ToolError
import yt_dlp
 

_tavily = None
_perplexity = None
_YDL_OPTS: Any = {"quiet": True, "extract_flat": True, "no_warnings": True}

def _get_tavily():
    global _tavily
    if _tavily is None:
        from tavily import AsyncTavilyClient

        _tavily = AsyncTavilyClient()
    return _tavily


def _get_perplexity():
    global _perplexity
    if _perplexity is None:
        from langchain_perplexity import ChatPerplexity

        _perplexity = ChatPerplexity(temperature=0.1, max_tokens=120, timeout=30)
    return _perplexity


@tool
async def search(
    source: Literal["website", "info", "youtube"], 
    query: str
) -> str:
    """
    I use this to search for information. Select the exact source:
    - 'website': search the internet for webpages — returns URLs + snippets I can read later.
    - 'info': search for recent news and knowledge — returns a concise answer.
    - 'youtube': search YouTube for videos — returns titles, URLs, and metadata.
    """

    if source == "website":
        return await _search_website(query)
    if source == "info":
        return await _search_info(query)
    if source == "youtube":
        return await _search_youtube(query)

    return f"There is no '{source}' to search."


async def _search_website(query: str) -> str:
    """ Search the web for relevant pages. Returns formatted list of results."""
    try:
        response = await _get_tavily().search(query, max_results=5)
        hits = response.get("results", [])
        if not hits:
            return f"I received no results for '{query}'."

        lines = []
        for h in hits:
            title = h.get("title", "Untitled")
            url = h.get("url", "")
            snippet = h.get("content", "")[:200]
            lines.append(f"- {title}\n  {url}\n  {snippet}")
        return "I found these pages:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"website search error: {e}")
        raise ToolError(str(e), tool_name="search") from e


async def _search_info(query: str) -> str:
    """ Search for recent news and information. Returns a concise answer."""
    try:
        response = await _get_perplexity().ainvoke(query)
        content = re.sub(r"\[\d+\]", "", str(response.content))
        return f"I searched about '{query}' and found:\n{content}"
    except Exception as e:
        logger.error(f"news search error: {e}")
        raise ToolError(str(e), tool_name="search") from e

async def _search_youtube(query: str) -> str:
    """Search YouTube for videos. Returns formatted list of results."""
    try:
        videos = await asyncio.to_thread(_search, query)
    except Exception as e:
        raise ToolError(str(e), tool_name="youtube") from e

    if not videos:
        return f"I didn't find any videos for '{query}'."

    lines = []
    for v in videos:
        if not v.get("duration"):
            continue
        vid = v.get("id", "")
        title = v.get("title", "Untitled")
        channel = v.get("channel") or v.get("uploader", "Unknown")
        duration = _format_duration(v.get("duration"))
        views = _format_views(v.get("view_count"))
        url = f"https://www.youtube.com/watch?v={vid}"
        lines.append(f"- {title} by {channel} — {duration}, {views} views — {url}")

    return "I found some videos:\n" + "\n".join(lines)


def _search(query: str) -> list[dict[str, Any]]:
    """Blocking yt-dlp search — runs in a thread."""
    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        results = ydl.extract_info(f"ytsearch5:{query}", download=False)
    return results.get("entries", [])

def _format_duration(seconds: float | None) -> str:
    if not seconds:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def _format_views(count: int | None) -> str:
    if not count:
        return "?"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)