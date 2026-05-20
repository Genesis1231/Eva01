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
from typing import Callable, Awaitable, Any

from config import logger


@dataclass
class ActionEvent:
    """A single outgoing action command."""
    type: str
    content: str | None = None
    metadata: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
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
        self._handlers: dict[str, list[ActionHandler]] = defaultdict(list)
        self._running = False

    # ── Registration (called at startup by action consumers) ──────────────

    def on(self, action_type: str, handler: ActionHandler) -> None:
        """Register an async handler for an action type.

        Multiple handlers per type are supported — all will be called
        in registration order when an event of that type arrives.
        """
        
        if handler in self._handlers[action_type]:
            logger.warning(f"ActionBuffer: handler already registered for <{action_type}>.")
            return
        try:
            self._handlers[action_type].append(handler)
        except Exception as e:
            logger.error(f"ActionBuffer: error registering handler for <{action_type}> — {e}")

    def off(self, action_type: str, handler: ActionHandler) -> None:
        """Unregister a handler for an action type if it is currently registered."""
        
        handlers = self._handlers.get(action_type)
        if not handlers:
            logger.warning(f"ActionBuffer: no handlers found for <{action_type}>, cannot unregister.")
            return

        try:
            handlers.remove(handler)
        except Exception as e:
            logger.error(f"ActionBuffer: error unregistering handler for <{action_type}> — {e}")
            return
        
        if not handlers:
            self._handlers.pop(action_type, None)   

    # ── Producer side (async — called by LangGraph tools) ────────────────

    async def put(
        self,
        action_type: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Push an action event onto the bus."""
        event = ActionEvent(
            type=action_type,
            content=content,
            metadata=metadata,
        )
        await self._queue.put(event)
        # logger.debug(f"ActionBuffer: put <{action_type}> — {str(content)}")

    # ── Dispatch loop (async — runs as a concurrent task in the spine) ──

    async def start_loop(self) -> None:
        """Dispatch events to registered handlers. Runs forever."""
        self._running = True
        
        # Fun startup action to confirm we're alive
        # TODO: dynamic greeting based on previous interactions, time of day, etc.
        await self.put("speak", datetime.now().strftime("%A, %B %d, %Y"))
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

                for handler in tuple(handlers):
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

    def empty(self) -> bool:
        """True when the queue is empty."""
        return self._queue.empty()
