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
from typing import Optional

from .speaker import Speaker
from .audio_player import AudioPlayer
from ..action_buffer import ActionBuffer, ActionEvent


class VoiceActor:
    """
    Two-channel audio actor: speech and music run independently.
    Registers as handler on ActionBuffer for speak/interrupt events.
    """

    def __init__(self, speaker: Speaker):
        self.speaker = speaker or Speaker()
        self.music_player = AudioPlayer()

        self.current_speech_task: Optional[asyncio.Task] = None
        self.current_music_task: Optional[asyncio.Task] = None
        self.is_speaking: bool = False

    def register(self, buffer: ActionBuffer) -> None:
        """Register speak/interrupt handlers on the action buffer."""
        buffer.on("speak", self._handle_speak)
        buffer.on("interrupt", self._handle_interrupt)

    async def _handle_speak(self, event: ActionEvent) -> None:
        """Handle speak: cancel current speech and start new one."""
        if not event.content:
            return

        await self._cancel_speech()

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

    async def play_music(self, url: str) -> None:
        """Start/replace background music. Does not interrupt speech."""
        await self.stop_music()
        self.current_music_task = asyncio.create_task(self._music_loop(url))

    async def stop_music(self) -> None:
        """Stop background music only. Does not interrupt speech."""
        if self.current_music_task and not self.current_music_task.done():
            self.current_music_task.cancel()
            try:
                await self.current_music_task
            except asyncio.CancelledError:
                pass
            self.current_music_task = None

    async def _music_loop(self, url: str) -> None:
        """Runs music playback in a thread. Cleans up mpv on cancel."""
        try:
            await asyncio.to_thread(self.music_player.play_stream, url)
        except asyncio.CancelledError:
            self.music_player.stop_playback()
            raise

    async def stop(self):
        """Stop all audio channels."""
        logger.debug("Voice Actor stopping...")
        await self._cancel_speech()
        await self.stop_music()
        logger.debug("Voice Actor stopped.")
