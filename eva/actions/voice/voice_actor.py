"""
VoiceActor:
    EVA consumes speak/interrupt from ActionBuffer, plays via Speaker.
    Two independent audio channels: speech (via Speaker) and music (via AudioPlayer).
    
    start_loop() -> runs forever, blocks
    stop() -> cancels loop and stops all audio
    play_music(url) -> start/replace background music (does not interrupt speech)
    stop_music() -> stop background music only
"""

import asyncio
from config import logger
from typing import Optional

from .speaker import Speaker
from .audio_player import AudioPlayer
from ..action_buffer import ActionBuffer

class VoiceActor:
    """
    Two-channel audio actor: speech and music run independently.

    Attributes:
        buffer: The action buffer to consume commands from.
        speaker: The speech channel (TTS model + its own AudioPlayer).
        music_player: The music channel (dedicated AudioPlayer, mpv only).
        current_speech_task: Tracked so speech can be interrupted without touching music.
        current_music_task: Tracked so music can be stopped without touching speech.
    """

    def __init__(self, action_buffer: ActionBuffer, speaker: Speaker):
        self.buffer = action_buffer
        self.speaker = speaker or Speaker()
        self.music_player = AudioPlayer()          # dedicated music channel

        # Tasks
        self.current_speech_task: Optional[asyncio.Task] = None
        self.current_music_task: Optional[asyncio.Task] = None
        self._loop_task: Optional[asyncio.Task] = None

        self._running = False
        self.is_speaking: bool = False

    async def start_loop(self):
        """Start the voice actor loop."""
        self._loop_task = asyncio.current_task()
        self._running = True
        logger.debug("Voice Actor started.")
        
        while self._running:
            try:
                command = await self.buffer.get()
                await self._handle_command(command)
                        
            except asyncio.CancelledError:
                logger.debug("Voice Actor loop cancelled.")
                self._running = False
                await self._cancel_speech()
                break
            
            except Exception as e:
                logger.error(f"Voice Actor error: {e}")
    
    async def _handle_command(self, command):
        """Dispatch command to appropriate handler."""
        if command.type == "speak" and command.content:
            await self._handle_speak(command)
        elif command.type == "interrupt":
            await self._handle_interrupt()

    async def _handle_speak(self, command):
        """Handle speak command: cancel current speech and start new one."""
        await self._cancel_speech()

        language = command.metadata.get("language", "en")
        self.is_speaking = True

        # Run blocking speak in a thread so the event loop stays free
        self.current_speech_task = asyncio.create_task(
            asyncio.to_thread(self.speaker.speak, command.content, language)
        )
        self.current_speech_task.add_done_callback(
            lambda _: setattr(self, 'is_speaking', False)
        )

    async def _handle_interrupt(self):
        """Handle interrupt command: stop current speech."""
        if self.current_speech_task and not self.current_speech_task.done():
            await self._cancel_speech()
            logger.info("Voice actor interrupted speech.")

    async def _cancel_speech(self):
        """Cancel current speech task and stop speaker output."""
        if self.current_speech_task and not self.current_speech_task.done():
            self.speaker.stop_speaking()       # unblock sd.wait() / mpv
            try:
                await self.current_speech_task  # wait for thread to actually finish
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
            # Wait for task to actually cancel to ensure cleanup
            try:
                await self.current_music_task
            except asyncio.CancelledError:
                pass
            self.current_music_task = None

    async def _music_loop(self, url: str) -> None:
        """Runs music playback in a thread. Cleans up mpv on cancel."""
        try:
            # Fixed: AudioPlayer has play_stream, not stream
            await asyncio.to_thread(self.music_player.play_stream, url)
        except asyncio.CancelledError:
            self.music_player.stop_playback()
            raise

    async def stop(self):
        """Stop the Voice Actor and all audio channels."""
        self._running = False
        logger.debug("Voice Actor stopping...")

        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()

        await self._cancel_speech()
        await self.stop_music()
        logger.debug("Voice Actor stopped.")
