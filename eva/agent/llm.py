"""
Cortex: EVA's language faculty.

Owns the LLM and prompt construction.
Brain calls cortex.respond(messages) — everything else is internal.
"""

from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    BaseMessage,
    trim_messages
)

from config import logger
from eva.agent.constructor import PromptConstructor
from eva.subconscious.mood import render_mood


class Cortex:
    """EVA's language cortex — wraps LLM + prompt into a single respond() call."""

    _TEMPERATURE = 0.8  # creative but not too random
    _MAX_HISTORY_TOKENS = 8000  # rough limit for context window, leaving room for response

    def __init__(
        self,
        model_name: str,
        tools: list[BaseTool],
    ) -> None:

        self.model_name = model_name
        self._llm = init_chat_model(
            model=model_name,
            temperature=self._TEMPERATURE
        ).bind_tools(tools)

        logger.debug(f"Cortex: {model_name} ready with {len(tools)} tools.")

    async def respond(
        self,
        constructor: PromptConstructor,
        messages: list[BaseMessage],
        present_people: set[str],
        memory: str = "",
        mood: list[float] | None = None,
    ) -> AIMessage:
        """Construct prompt, trim messages, invoke LLM, return response."""

        # if sense is audio text, add to messages as HumanMessage;
        # if it's text, add to system prompt as OBSERVATION
        timestamp = datetime.now().strftime("%A, %B %d, %Y at %I%p")
        system = constructor.build_system(
            timestamp=timestamp,
            memory=memory,
            present_people=present_people,
            mood_block=render_mood(mood),
        )

        # trim to fit context window, keeping recent messages
        messages = trim_messages(messages, max_tokens=self._MAX_HISTORY_TOKENS, token_counter='approximate')  
        
        # Only add the kickoff prompt on the initial pass, not on ReAct continuations
        # (where the last message is a ToolMessage from a prior tool call)
        complete_prompt = [SystemMessage(content=system)] + messages
                         
        # logger.debug(f"Cortex received messages:\n{complete_prompt}\n")

        try:
            response = await self._llm.ainvoke(complete_prompt)
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

        logger.debug(f"Cortex response: {response.content}\n")
        return response
