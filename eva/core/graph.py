"""
EVA's brain — a LangGraph StateGraph with ReAct tool loop.

Graph: START → think → tool calls? → yes → tools → think
                                    → no  → END

Pure workflow topology. The ChatAgent owns the LLM and prompt logic.
"""

from datetime import datetime
from typing import List, Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from eva.agent.chatagent import ChatAgent
from eva.senses.sense_buffer import SenseEntry


class EvaState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    present_people: List[str]


class Brain:
    """EVA's brain graph — topology only, agent owns the process."""

    def __init__(self, agent: ChatAgent, checkpointer=None):
        self.agent = agent
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

        async def think(state: EvaState):
            """ The "think" node — EVA processes messages and decides on tool calls."""
            response = await agent.think(
                state["messages"],
                present_people=state.get("present_people", []),
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
        prefix = "I hear: " if entry.type == "audio" else "I see: "
        content = prefix + entry.content

        # Extract face IDs from vision metadata
        face_ids = []
        if entry.metadata and "faces" in entry.metadata:
            face_ids = entry.metadata["faces"]

        # Core orchestration owns state updates for seen people.
        people_db = getattr(self.agent.constructor, "people_db", None)
        if people_db and face_ids:
            for face_id in set(face_ids):
                await people_db.touch(face_id)

        await self._graph.ainvoke(
            {
                "messages": [HumanMessage(content=content)], 
                "present_people": face_ids
            },
            config=self._config,
        )
