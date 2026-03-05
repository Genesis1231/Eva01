"""
ActionBuffer — Outgoing event bus for EVA's actions.

Mirrors SenseBuffer but flows in the opposite direction:
    Brain/Tools  →  ActionBuffer  →  registered handlers (voice, screen, etc.)

Tools push events:       await buffer.put("speak", text)
Consumers register:      buffer.on("speak", handler)
Spine runs the loop:     await buffer.start_loop()
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Awaitable, Optional, Dict, Any, List

from config import logger


@dataclass
class ActionEvent:
    """A single outgoing action command."""
    type: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the action event to a dictionary."""
        d = {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# Handler signature: async def handler(event: ActionEvent) -> None
ActionHandler = Callable[[ActionEvent], Awaitable[None]]


class ActionBuffer:
    """
    Async event bus — tools push, registered handlers consume.

    Attributes:
        _queue:    Async queue for action events.
        _handlers: Map of action type → list of async handler functions.
        _running:  Whether the dispatch loop is active.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ActionEvent] = asyncio.Queue()
        self._handlers: Dict[str, List[ActionHandler]] = defaultdict(list)
        self._running = False

    # ------------------------------------------------------------------
    # Registration (called at startup by action consumers)
    # ------------------------------------------------------------------

    def on(self, action_type: str, handler: ActionHandler) -> None:
        """Register an async handler for an action type.

        Multiple handlers per type are supported — all will be called
        in registration order when an event of that type arrives.
        """
        self._handlers[action_type].append(handler)
        logger.debug(f"ActionBuffer: registered handler for <{action_type}>")

    # ------------------------------------------------------------------
    # Producer side (async — called by LangGraph tools)
    # ------------------------------------------------------------------

    async def put(
        self,
        action_type: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Push an action event onto the bus."""
        event = ActionEvent(
            type=action_type,
            content=content,
            metadata=metadata,
        )
        await self._queue.put(event)
        logger.debug(f"ActionBuffer: put <{action_type}> — {len(str(content).split())} words.")

    # ------------------------------------------------------------------
    # Dispatch loop (async — runs as a concurrent task in the spine)
    # ------------------------------------------------------------------

    async def start_loop(self) -> None:
        """Dispatch events to registered handlers. Runs forever."""
        self._running = True
        logger.debug("ActionBuffer: dispatch loop started.")

        while self._running:
            try:
                event = await self._queue.get()

                if not self._running:
                    break

                handlers = self._handlers.get(event.type)

                if not handlers:
                    logger.warning(f"ActionBuffer: no handler for <{event.type}>, dropped.")
                    continue

                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"ActionBuffer: handler error for <{event.type}> — {e}")

            except asyncio.CancelledError:
                logger.debug("ActionBuffer: dispatch loop cancelled.")
                self._running = False
                break

    async def stop(self) -> None:
        """Stop the dispatch loop."""
        self._running = False
        # Unblock the queue.get() so the loop can exit
        await self._queue.put(ActionEvent(type="_stop"))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def empty(self) -> bool:
        """True when the queue is empty."""
        return self._queue.empty()
