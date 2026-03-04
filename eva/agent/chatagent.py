"""
ChatAgent: EVA's language model interface.

Owns the LLM, tools, and prompt construction.
The graph calls agent.think(messages) — everything else is internal.
"""

from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage, AIMessage, ToolMessage, HumanMessage, trim_messages,
)

from config import logger
from eva.actions.action_buffer import ActionBuffer
from eva.agent.constructor import PromptConstructor
from eva.tools import load_tools


class ChatAgent:
    """EVA's brain process — wraps LLM + tools + prompt into a single think() call."""

    def __init__(self, model_name: str, action_buffer: ActionBuffer) -> None:
        self.model_name = model_name
        self.constructor = PromptConstructor()
        self.tools = load_tools(action_buffer)
        self._llm = init_chat_model(model_name, temperature=0.8).bind_tools(self.tools)

        logger.debug(f"ChatAgent: {model_name} ready with {len(self.tools)} tools.")

    async def think(self, messages: list) -> AIMessage:
        """Distill history, trim, inject system prompt, invoke LLM."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        system = self.constructor.build_system(timestamp)
        distilled = self._distill_history(messages)
        trimmed = trim_messages(distilled, max_tokens=8000, token_counter='approximate')

        # logger.debug(f"History: {len(messages)} raw → {len(distilled)} distilled → {len(trimmed)} trimmed")

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

    @staticmethod
    def _distill_history(messages: list) -> list:
        """Collapse completed feel/speak tool cycles into clean AIMessages.

        Only distills PREVIOUS turns (before the last HumanMessage).
        The current turn stays raw so the ReAct loop can continue.
        """
        # Find the last HumanMessage — everything after it is the current turn
        last_human_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break

        # Distill only the history portion (before last HumanMessage)
        history = messages[:last_human_idx] if last_human_idx > 0 else []
        current_turn = messages[last_human_idx:] if last_human_idx >= 0 else messages[:]

        result = []
        i = 0

        while i < len(history):
            msg = history[i]

            # Only try to distill AIMessages that have tool_calls
            if not isinstance(msg, AIMessage) or not getattr(msg, 'tool_calls', None):
                # Drop empty AIMessages left over from completed cycles
                if isinstance(msg, AIMessage) and not msg.content and not getattr(msg, 'tool_calls', None):
                    i += 1
                    continue
                result.append(msg)
                i += 1
                continue

            # Check if ALL tool calls are feel/speak (distillable)
            tool_calls = msg.tool_calls
            tool_names = {tc['name'] for tc in tool_calls}
            distillable = tool_names <= {'feel', 'speak'}

            if not distillable:
                result.append(msg)
                i += 1
                continue

            # Check that all matching ToolMessages exist (cycle is complete)
            call_ids = {tc['id'] for tc in tool_calls}
            tool_msg_count = 0
            j = i + 1
            while j < len(history) and isinstance(history[j], ToolMessage):
                if history[j].tool_call_id in call_ids:
                    tool_msg_count += 1
                j += 1

            if tool_msg_count < len(call_ids):
                result.append(msg)
                i += 1
                continue

            # Distill the completed cycle into a single AIMessage
            parts = []
            for tc in tool_calls:
                name = tc['name']
                args = tc['args']
                if name == 'feel':
                    feeling = args.get('feeling', '')
                    monologue = args.get('inner_monologue', '')
                    parts.append(f"[I felt {feeling}]")
                elif name == 'speak':
                    text = args.get('text', '')
                    parts.append(f'I said: "{text}"')

            result.append(AIMessage(content="\n\n".join(parts)))

            # Skip past the ToolMessages (and any trailing empty AIMessage)
            i = j
            if i < len(history) and isinstance(history[i], AIMessage) and not history[i].content and not getattr(history[i], 'tool_calls', None):
                i += 1

        # Append current turn unchanged
        result.extend(current_turn)
        return result
