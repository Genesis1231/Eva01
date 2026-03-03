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
from ..buffer import ActionBuffer

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

    def __init__(self, action_buffer: ActionBuffer):
        self.buffer = action_buffer
        self.speaker = Speaker()
        self.music_player = AudioPlayer()          # dedicated music channel
        
        # Tasks
        self.current_speech_task: Optional[asyncio.Task] = None
        self.current_music_task: Optional[asyncio.Task] = None
        self._loop_task: Optional[asyncio.Task] = None
        
        self._running = False

    async def start_loop(self):
        """Start the voice actor loop."""
        self._loop_task = asyncio.current_task()
        self._running = True
        logger.debug("Voice Actor started.")
        
        while self._running:
            try:
                command = await self.buffer.get()
                
                if command.type == "speak" and command.content:
                    # Cancel any existing speech
                    if self.current_speech_task and not self.current_speech_task.done():
                        self.current_speech_task.cancel()
                        await self.speaker.stop_speaking()
                    
                    language = command.metadata.get("language", "en")
                    # Launch new speech in a background task
                    self.current_speech_task = asyncio.create_task(
                        self.speaker.speak(command.content, language)
                    )

                elif command.type == "interrupt":
                    # Instantly stop speaking
                    if self.current_speech_task and not self.current_speech_task.done():
                        self.current_speech_task.cancel()
                        if hasattr(self.speaker, 'stop_speaking'):
                            await self.speaker.stop_speaking()
                        logger.info("Voice actor interrupted speech.")
                        
            except asyncio.CancelledError:
                logger.debug("Voice Actor loop cancelled.")
                self._running = False
                if self.current_speech_task and not self.current_speech_task.done():
                    self.current_speech_task.cancel()
                break
            
            except Exception as e:
                logger.error(f"Voice Actor error: {e}")
    
                
    async def play_music(self, url: str) -> None:
        """Start/replace background music. Does not interrupt speech."""
        await self.stop_music()
        self.current_music_task = asyncio.create_task(self._music_loop(url))

    async def stop_music(self) -> None:
        """Stop background music only. Does not interrupt speech."""
        if self.current_music_task and not self.current_music_task.done():
            self.current_music_task.cancel()
            self.current_music_task = None

    async def _music_loop(self, url: str) -> None:
        """Runs music playback in a thread. Cleans up mpv on cancel."""
        try:
            await asyncio.to_thread(self.music_player.stream, url)
        except asyncio.CancelledError:
            self.music_player.stop_playback()
            raise

    async def stop(self):
        """Stop the Voice Actor and all audio channels."""
        self._running = False
        logger.debug("Voice Actor stopped.")

        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()

        if self.current_speech_task and not self.current_speech_task.done():
            self.current_speech_task.cancel()
            await self.speaker.stop_speaking()

        await self.stop_music()
