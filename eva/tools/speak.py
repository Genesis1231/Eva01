"""EVA's voice tool — pushes speech to ActionBuffer."""

from langchain_core.tools import tool
from eva.actions.action_buffer import ActionBuffer
from eva.utils.feed import feed_post


def make_speak_tool(action_buffer: ActionBuffer):
    """Create a speak tool bound to the given ActionBuffer."""

    @tool
    async def speak(text: str) -> str:
        """Say something out loud. I use this to speak."""
        await action_buffer.put("speak", text)
        feed_post("speech", text)  # her voice → the Room stream
        return text

    return speak
