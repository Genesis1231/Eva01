"""
ElevenLabsSpeaker: ElevenLabs TTS model.
  eva_speak(text, language) -> streams via elevenlabs stream(), runs in thread
  generate_audio(text, language, media_folder) -> writes mp3, returns relative path
  stop_playback() -> stops the audio playback.
"""

from config import logger
import os
import tempfile
import asyncio
import secrets
from typing import Optional

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings, stream
from .audio_player import AudioPlayer
    
class ElevenLabsSpeaker:
    """Audio speaker using ElevenLabs TTS."""

    def __init__(self, voice: str = "TbMNBJ27fH2U0VgpSNko") -> None:
        self.model: ElevenLabs = ElevenLabs()
        self.audio_player: AudioPlayer = AudioPlayer()
        self.voice: str = voice # voice could be configured in the future
        
    def eva_speak(self, text: str, language: Optional[str] = None) -> None:
        """Speak the given text using ElevenLabs. Blocking — run via to_thread."""
        model_name = "eleven_flash_v2" if language == "en" else "eleven_v3"

        try:
            audio_stream = self.model.text_to_speech.stream(
                model_id=model_name,
                output_format="mp3_22050_32",
                text=text,
                voice_id=self.voice,
                optimize_streaming_latency=1,
            )
            stream(audio_stream)

        except Exception as e:
            logger.error(f"Error during ElevenLabs synthesis: {e}")
            
    async def generate_audio(self, text: str, language: Optional[str], media_folder: str) -> Optional[str]:
        """ Generate mp3 from text using ElevenLabs """
        
        model_name: str = "eleven_flash_v2" if language == "en" else "eleven_v3"
        
        filename = f"{secrets.token_hex(16)}.mp3"
        file_path = os.path.join(media_folder, "audio", filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        def _generate():
            audio_stream = self.model.text_to_speech.convert(
                model_id=model_name,
                output_format="mp3_22050_32",                      
                text=text,
                voice_id=self.voice,
                optimize_streaming_latency = 1,
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.8,
                    use_speaker_boost=True
                )
            )
            
            with open(file_path, 'wb') as f:
                f.write(b"".join(audio_stream))

        try:
            await asyncio.to_thread(_generate)
            # Log the file path for debugging
            logger.debug(f"Audio file saved to: {file_path}")
            
            return f"audio/{filename}"
        
        except Exception as e:
            logger.error(f"Error during ElevenLabs synthesis: {e}")
            return None

    def stop_playback(self) -> None:
        """Stop the audio playback. Thread-safe."""
        if hasattr(self, 'audio_player'):
            self.audio_player.stop_playback()
