"""
VoiceActor:
    EVA's voice — handles speak/interrupt actions from ActionBuffer.
    Two independent audio channels: speech (via Speaker) and music (via AudioPlayer).

    register(buffer) -> registers speak/interrupt handlers on ActionBuffer
    stop() -> stops all audio
    play_music(url) -> start/replace background music (does not interrupt speech)
    stop_music() -> stop background music only
"""

import asyncio
from config import logger

from .speaker import Speaker
from ..action_buffer import ActionBuffer, ActionEvent
from ..base import BaseAction


class VoiceActor(BaseAction):
    """
    Two-channel audio actor: speech and music run independently.
    Registers as handler on ActionBuffer for speak/interrupt events.
    """

    def __init__(self, speaker: Speaker):
        self.speaker = speaker or Speaker()
        self.current_speech_task: asyncio.Task | None = None
        self.is_speaking: bool = False

    def register(self, buffer: ActionBuffer) -> None:
        """Register speak/interrupt handlers on the action buffer."""
        buffer.on("speak", self._handle_speak)
        buffer.on("interrupt", self._handle_interrupt)
        
    async def _handle_speak(self, event: ActionEvent) -> None:
        """Handle speak: wait for current speech to finish, then speak next."""
        if not event.content:
            return

        # Queue: wait for previous speech to finish instead of cancelling
        if self.current_speech_task and not self.current_speech_task.done():
            try:
                await self.current_speech_task
            except (asyncio.CancelledError, Exception):
                pass

        language = (event.metadata or {}).get("language", "en")
        self.is_speaking = True

        self.current_speech_task = asyncio.create_task(
            asyncio.to_thread(self.speaker.speak, event.content, language)
        )
        self.current_speech_task.add_done_callback(
            lambda _: setattr(self, 'is_speaking', False)
        )

    async def _handle_interrupt(self, event: ActionEvent) -> None:
        """Handle interrupt: stop current speech."""
        if self.current_speech_task and not self.current_speech_task.done():
            await self._cancel_speech()
            logger.debug("Voice actor interrupted speech.")

    async def _cancel_speech(self):
        """Cancel current speech task and stop speaker output."""
        if self.current_speech_task and not self.current_speech_task.done():
            self.speaker.stop_speaking()
            try:
                await self.current_speech_task
            except Exception:
                pass
        self.current_speech_task = None
        self.is_speaking = False

    async def start(self) -> None:
        """ No background tasks to start for VoiceActor"""
        pass
    
    async def stop(self):
        """Stop all audio channels and release models."""
        await self._cancel_speech()
        self.speaker.close()
        logger.debug("Voice Actor stopped.")
