"""
EVA's brain — a LangGraph StateGraph with ReAct tool loop.

Graph: START → think → tool calls? → yes → tools → think
                                    → no  → END

Pure workflow topology. The ChatAgent owns the LLM and prompt logic.
"""

from datetime import datetime
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from eva.agent.chatagent import ChatAgent


class Brain:
    """EVA's brain graph — topology only, agent owns the process."""

    def __init__(self, agent: ChatAgent, checkpointer=None):
        self.agent = agent
        self.thread_id = self._new_thread_id()
        self._config = {"configurable": {"thread_id": self.thread_id}}
        self._graph = self._build(checkpointer)

    def _new_thread_id(self) -> str:
        """Generate a new thread ID."""
        return f"eva-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def _build(self, checkpointer):
        agent = self.agent

        async def think(state: MessagesState):
            response = await agent.think(state["messages"])
            return {"messages": [response]}

        def route(state: MessagesState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "__end__"

        builder = StateGraph(MessagesState)
        builder.add_node("think", think)
        builder.add_node("tools", ToolNode(agent.tools))

        builder.set_entry_point("think")
        builder.add_conditional_edges("think", route)
        builder.add_edge("tools", "think")

        return builder.compile(checkpointer=checkpointer)

    async def get_messages(self) -> list:
        """Read current message history from the checkpointer."""
        state = await self._graph.aget_state(self._config)
        if state and state.values:
            return state.values.get("messages", [])
        return []

    async def invoke(self, sense: str):
        """Send a sensory input through the graph."""
        await self._graph.ainvoke(
            {"messages": [HumanMessage(content=sense)]},
            config=self._config,
        )
