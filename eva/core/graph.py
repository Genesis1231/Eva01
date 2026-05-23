"""
EVA's brain — a LangGraph StateGraph with ReAct tool loop.

Graph: START → think → tool calls? → yes → tools → think
                                    → no  → END

Pure workflow topology. The Cortex owns the LLM and prompt logic.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Annotated, TypedDict, Set

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage
)

from config import logger
from eva.actions.action_buffer import ActionBuffer
from eva.agent.llm import Cortex
from eva.agent.constructor import PromptConstructor
from eva.core.memory import MemoryDB
from eva.senses.sense_buffer import SenseEntry
from eva.subconscious.mood import MoodScorer, _update_mood
from eva.tools import load_tools, handle_tool_error


class EvaState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    present_people: Set[str]
    mood: Annotated[List[float], _update_mood]

class Brain:
    """EVA's brain graph — orchestrates agent, memory, and workflow. """
    
    _RECURSION_LIMIT = 50

    def __init__(
        self,
        model_name: str,
        action_buffer: ActionBuffer,
        memory: MemoryDB,
        people: Dict,        
        checkpointer=None,
    ):
        self.tools = load_tools(action_buffer)
        self.cortex = Cortex(model_name=model_name, tools=self.tools)
        self.constructor = PromptConstructor(people=people)
        self.memory = memory
        self.mood_scorer = MoodScorer()
        
        self._processing = False
        self.thread_id = self._new_thread_id()
        self._config = self._get_config()
        self._terminal_tools = {t.name for t in self.tools if (t.metadata or {}).get("terminal")}
        self._graph = self._build(checkpointer)

    def _new_thread_id(self) -> str:
        """Generate a new thread ID."""
        return f"eva-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def _get_config(self) -> RunnableConfig:
        """Get the current config for graph execution."""
        return RunnableConfig(
            configurable={"thread_id": self.thread_id},
            recursion_limit=self._RECURSION_LIMIT,
        )
    
    async def _think(self, state: EvaState):
        """ The "think" node — EVA processes messages and decides on tool calls."""

        distilled, memory = await self.memory.prepare_context(state["messages"])

        response = await self.cortex.respond(
            constructor=self.constructor,
            messages=distilled,
            present_people=state.get("present_people", set()),
            memory=memory,
            mood=state.get("mood"),
        )

        return {"messages": [response]}

    def _route(self, state: EvaState):
        """After thinking, route to tools if tool calls exist, otherwise end."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def _tool_route(self, state: EvaState):
        """After tools execute, route based on tool type.

        Terminal tools (stay_quiet) → END (explicit choice to stop).
        Everything else (feel, speak, etc.) → think (ReAct loop continues).
        """
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                called = {tc["name"] for tc in msg.tool_calls}
                if called <= self._terminal_tools:
                    return END
                return "think"
        return END

    def _build(self, checkpointer):
        """ Build the StateGraph for EVA's brain."""
        builder = StateGraph(EvaState)

        builder.add_node("think", self._think)
        builder.add_node("tools", ToolNode(
            self.tools,
            handle_tool_errors=handle_tool_error,
        ))

        builder.set_entry_point("think")
        builder.add_conditional_edges("think", self._route)
        builder.add_conditional_edges("tools", self._tool_route)

        return builder.compile(checkpointer=checkpointer)


    def is_busy(self) -> bool:
        """ Indicates if the brain is currently processing an input."""
        return self._processing

    async def invoke(self, entry: SenseEntry):
        """Send a sensory input through the graph."""
        self._processing = True
        try:
            # Extract people IDs from vision and audio metadata
            people_ids = set()
            if entry.metadata:
                if "faces" in entry.metadata:
                    people_ids.update(entry.metadata["faces"])
                if "speaker_id" in entry.metadata:
                    people_ids.add(entry.metadata["speaker_id"])

            # Track seen people for relationship reflection at flush time.
            if people_ids:
                self.memory.add_people_to_session(people_ids)

            message = HumanMessage(content=entry.content)

            # External senses drive mood; inner-voice "thought" events
            # are excluded — outside factors change mood, not narration.
            state_update: dict = {
                "messages": [message],
                "present_people": people_ids,
            }
            if entry.type != "thought":
                state_update["mood"] = await asyncio.to_thread(
                    self.mood_scorer.score, entry.content
                )

            await self._graph.ainvoke(state_update, config=self._config)
        finally:
            self._processing = False

    async def shutdown(self):
        """Graceful shutdown — flush memory, close resources."""
        
        state = await self._graph.aget_state(config=self._config)
        
        if not state or not state.values:
            logger.debug("Brain shutdown: no state found, skipping memory flush.")
            return
        
        messages = state.values.get("messages", [])
        if messages:
            try:
                await self.memory.flush(messages, session_id=self.thread_id)
            except Exception as e:
                logger.error(f"EVA: failed to flush memory — {e}")
