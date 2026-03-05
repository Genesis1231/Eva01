"""
EdgeSpeaker: Edge TTS model.
  eva_speak(text, language) -> synthesizes to temp file, plays via AudioPlayer
  generate_audio(text, language, media_folder) -> writes mp3, returns relative path
  stop_playback() -> stops AudioPlayer
"""

import asyncio
import os
import secrets
import tempfile
from typing import Optional

import edge_tts
from config import logger

from .audio_player import AudioPlayer

# Maps language codes to the best default Edge TTS voice for that language.
# Run `edge-tts --list-voices` to see all available options.
# LANGUAGE_VOICES: dict[str, str] = {
#     "en": "en-US-AriaNeural",
#     "zh": "zh-CN-XiaoxiaoNeural",
#     "ja": "ja-JP-NanamiNeural",
#     "ko": "ko-KR-SunHiNeural",
#     "fr": "fr-FR-DeniseNeural",
#     "de": "de-DE-KatjaNeural",
#     "es": "es-ES-ElviraNeural",
# }

DEFAULT_VOICE = "en-US-AriaNeural"

class EdgeSpeaker:
    """
    Edge TTS speaker.
    Supports multiple languages via a language-to-voice mapping.
    """

    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        self.voice = voice
        self.audio_player = AudioPlayer()

    def _voice_for(self, language: Optional[str]) -> str:
        """Get the appropriate voice for the given language."""
        # For now, just return the default voice
        # Could be extended with language-specific voice mapping
        return self.voice

    def eva_speak(self, text: str, language: Optional[str] = None) -> None:
        """Speak the given text using Edge TTS. Blocking — run via to_thread."""
        try:
            voice = self._voice_for(language)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name

            asyncio.run(edge_tts.Communicate(text, voice).save(temp_path))
            self.audio_player.play_stream(temp_path)

        except Exception as e:
            logger.error(f"Error during Edge TTS synthesis: {e}")

    async def generate_audio(self, text: str, language: Optional[str], media_folder: str) -> Optional[str]:
        """Generate mp3 from text and save to the media folder."""
        
        filename = f"{secrets.token_hex(16)}.mp3"
        file_path = os.path.join(media_folder, "audio", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            voice = self._voice_for(language)
            await edge_tts.Communicate(text, voice).save(file_path)
            
            logger.debug(f"Audio file saved to: {file_path}")
            return f"audio/{filename}"

        except Exception as e:
            logger.error(f"Error during Edge TTS synthesis: {e}")
            return None

    def stop_playback(self) -> None:
        """Stop audio playback. Thread-safe."""
        try:
            self.audio_player.stop_playback()
        except Exception:
            pass
