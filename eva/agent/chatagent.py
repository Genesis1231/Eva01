"""
ChatAgent: EVA's language model interface.

Owns the LLM, tools, and prompt construction.
The graph calls agent.think(messages) — everything else is internal.
"""

from datetime import datetime
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, trim_messages

from config import logger
from eva.actions.action_buffer import ActionBuffer
from eva.agent.constructor import PromptConstructor
from eva.tools import load_tools


class ChatAgent:
    """EVA's brain process — wraps LLM + tools + prompt into a single think() call."""

    def __init__(self, model_name: str, action_buffer: ActionBuffer, memory=None) -> None:
        self.model_name = model_name
        self.memory = memory
        self.constructor = PromptConstructor()
        self.tools = load_tools(action_buffer)
        self._llm = init_chat_model(model_name, temperature=0.8).bind_tools(self.tools)

        logger.debug(f"ChatAgent: {model_name} ready with {len(self.tools)} tools.")

    async def think(self, messages: list) -> AIMessage:
        """Distill history, trim, inject system prompt, invoke LLM."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if self.memory:
            distilled, journal = self.memory.prepare_context(messages)
            system = self.constructor.build_system(timestamp, memory=journal or None)
        else:
            distilled = messages
            system = self.constructor.build_system(timestamp)

        trimmed = trim_messages(distilled, max_tokens=8000, token_counter='approximate')

        response = await self._llm.ainvoke([SystemMessage(content=system)] + trimmed)

        usage = response.usage_metadata
        if usage:
            logger.debug(
                f"LLM({self.model_name}) — "
                f"input: {usage['input_tokens']/1000:.1f}k  "
                f"output: {usage['output_tokens']/1000:.1f}k  "
                f"total: {usage['total_tokens']/1000:.1f}k"
            )

        return response
