"""EVA's show tool — opens a URL on screen via the browser."""

from langchain_core.tools import tool
from typing import Literal
from eva.actions.action_buffer import ActionBuffer
from eva.utils.feed import feed_post


def make_show_tool(action_buffer: ActionBuffer):
    """Create a show tool bound to the given ActionBuffer."""

    @tool
    async def show(device: Literal["browser"] = "browser", url: str = "") -> str:
        """
        I use this to show something with my device.
        select 'browser' when I want to show a webpage, video, or anything with a URL.
        I am pretty careful with the URL I provide, since it will be opened directly.
        """

        if device == "browser":
            await action_buffer.put("show", url)
            feed_post("browser", url)  # the page on her desk → the Room canvas
            return f'I am opening the browser to show: {url}'

        return f"I don't know how to show things on '{device}', yet."

    return show
