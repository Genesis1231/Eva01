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
from eva.core.people import PeopleDB
from eva.core.memory import MemoryDB
from eva.tools import load_tools


class ChatAgent:
    """EVA's brain process — wraps LLM + tools + prompt into a single think() call."""
    
    _TEMPTURE = 0.8  # creative but not too random
    
    def __init__(
        self, 
        model_name: str, 
        action_buffer: ActionBuffer, 
        people_db: PeopleDB,
        memory: Optional[MemoryDB] = None,
    )-> None:
        
        self.model_name = model_name
        self.memory = memory
        self.constructor = PromptConstructor(people_db=people_db)
        self.tools = load_tools(action_buffer)
        self._llm = init_chat_model(
            model=model_name, 
            temperature=self._TEMPTURE
        ).bind_tools(self.tools)

        logger.debug(f"ChatAgent: {model_name} ready with {len(self.tools)} tools.")

    async def think(self, messages: list, present_people: list[str]) -> AIMessage:
        """Distill history, trim, inject system prompt, invoke LLM."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if self.memory:
            distilled, journal = await self.memory.prepare_context(messages)
            system = self.constructor.build_system(
                timestamp=timestamp, 
                memory=journal, 
                present_people=present_people
            )
        else:
            distilled = messages
            system = self.constructor.build_system(
                timestamp=timestamp, 
                present_people=present_people
            )

        # Optional: Cap the input tokens
        trimmed = trim_messages(distilled, max_tokens=8000, token_counter='approximate')

        try:
            response = await self._llm.ainvoke([SystemMessage(content=system)] + trimmed)
        except Exception as e:
            logger.error(f"LLM ainvoke failed: {e}")
            # Fallback to a safe AIMessage to prevent the agent from crashing
            response = AIMessage(content="[I am having trouble forming a coherent thought right now.]")

        # resource usage logging
        if usage := response.usage_metadata:
            logger.debug(
                f"LLM({self.model_name}) — "
                f"input: {usage.get('input_tokens', 0)/1000:.1f}k  "
                f"output: {usage.get('output_tokens', 0)/1000:.1f}k  "
                f"total: {usage.get('total_tokens', 0)/1000:.1f}k"
            )

        return response
