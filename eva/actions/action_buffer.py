"""
ActionBuffer: Outgoing event bus for EVA's actions. 
  put(action_type, content, metadata) -> None
    Actions: speak, interrupt
  get() -> ActionEvent
  empty() -> bool
"""

from config import logger
import asyncio
from dataclasses import dataclass, field
import time
from typing import Optional


@dataclass
class ActionEvent:
    """A single action command."""
    type: str              
    content: Optional[str] = None # the actual data (text to speak, etc.)
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ActionBuffer:
    """
    Async buffer for outgoing commands from LangGraph to the physical actions.
    """
    def __init__(self):
        self._queue: asyncio.Queue[ActionEvent] = asyncio.Queue()

    async def put(
        self, 
        action_type: str, 
        content: Optional[str] = None, 
        metadata: Optional[dict] = None
    ) -> None:
        """Push an action command to the actions."""
        event = ActionEvent(
            type=action_type,
            content=content,
            metadata=metadata or {}
        )
        await self._queue.put(event)
        logger.debug(f"ActionBuffer: Put <{action_type}> — {len(str(content).split())} words.")

    async def get(self) -> ActionEvent:
        """Wait for and retrieve the next action command."""
        return await self._queue.get()

    def empty(self) -> bool:
        """Check if the buffer is empty."""
        return self._queue.empty()
