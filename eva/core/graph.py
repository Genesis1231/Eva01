"""
EVA's brain — a LangGraph StateGraph with ReAct tool loop.

Graph: START → think → tool calls? → yes → tools → think
                                    → no  → END

Pure workflow topology. The Cortex owns the LLM and prompt logic.
"""

from datetime import datetime
from typing import List, Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from eva.agent.cortex import Cortex
from eva.core.memory import MemoryDB
from eva.senses.sense_buffer import SenseEntry


class EvaState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    present_people: List[str]


class Brain:
    """EVA's brain graph — orchestrates agent, memory, and workflow."""

    def __init__(self, agent: Cortex, memory: MemoryDB, checkpointer=None):
        self.agent = agent
        self.memory = memory
        self.thread_id = self._new_thread_id()
        self._config = self._get_config()
        self._graph = self._build(checkpointer)

    def _new_thread_id(self) -> str:
        """Generate a new thread ID."""
        return f"eva-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def _get_config(self) -> RunnableConfig:
        """Get the current config for graph execution."""
        return RunnableConfig(configurable={"thread_id": self.thread_id})
    
    def _build(self, checkpointer):
        """ Build the StateGraph for EVA's brain."""
        agent = self.agent
        memory = self.memory

        async def think(state: EvaState):
            """ The "think" node — EVA processes messages and decides on tool calls."""
            distilled, journal = await memory.prepare_context(state["messages"])
            response = await agent.respond(
                distilled,
                present_people=state.get("present_people", []),
                journal=journal,
            )
            return {"messages": [response]}

        def route(state: EvaState):
            """ Decide whether to route to tools."""
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls: 
                return "tools"
            return "__end__"

        builder = StateGraph(EvaState)
        builder.add_node("think", think)
        builder.add_node("tools", ToolNode(agent.tools))

        builder.set_entry_point("think")
        builder.add_conditional_edges("think", route)
        builder.add_edge("tools", "think")

        return builder.compile(checkpointer=checkpointer)

    async def get_messages(self) -> list:
        """Read current message history from the checkpointer."""
        
        state = await self._graph.aget_state(config=self._config)
        if state and state.values:
            return state.values.get("messages", [])
        return []

    async def invoke(self, entry: SenseEntry):
        """Send a sensory input through the graph."""
        content = entry.content

        # Extract face IDs from vision metadata
        face_ids = []
        if entry.metadata and "faces" in entry.metadata:
            face_ids = entry.metadata["faces"]

        # Track seen people for relationship reflection at flush time.
        if face_ids:
            self.memory.add_people_to_session(set(face_ids))

        await self._graph.ainvoke(
            {
                "messages": [HumanMessage(content=content)], 
                "present_people": face_ids
            },
            config=self._config,
        )
