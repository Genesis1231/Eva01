"""
KokoroSpeaker: Kokoro TTS model.
  eva_speak(text, language) -> generates PCM, plays via sounddevice
  generate_audio(text, language, media_folder) -> writes wav, returns relative path
  stop_playback() -> stops sounddevice
"""

import asyncio
import os
import secrets
from pathlib import Path
from io import BytesIO
from pydub import AudioSegment
from typing import Optional

import sounddevice as sd
import soundfile as sf
import numpy as np
from config import logger
from kokoro_onnx import Kokoro

from .audio_player import AudioPlayer

_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models" 
_LANG_MAP = {
    "en": "en-us", 
    "zh": "cmn", 
    "ja": "ja", 
    "fr": "fr-fr", 
    "it": "it", 
    "es": "es"
}

class KokoroSpeaker:
    """Local TTS using Kokoro (ONNX)."""

    def __init__(self, voice: str = "af_heart") -> None:
        onnx_path = _MODEL_DIR / "kokoro-v1.0.onnx"
        voices_path = _MODEL_DIR / "voices-v1.0.bin"

        if not onnx_path.exists() or not voices_path.exists():
            raise FileNotFoundError(f"Kokoro model files not found in {_MODEL_DIR}. ")
        
        self.voice = voice
        self.audio_player = AudioPlayer()
        self._model = Kokoro(str(onnx_path), str(voices_path))

    def _get_language(self, language: Optional[str]) -> str:
        return _LANG_MAP.get(language or "en", "en-us") if language else "en-us"

    def eva_speak(self, text: str, language: Optional[str] = None) -> None:
        """Speak the given text using Kokoro TTS. Blocking — run via to_thread."""
        try:
            samples, sample_rate = self._model.create(
                text=text,
                voice=self.voice,
                lang=self._get_language(language),
            )
            self.audio_player.play_pcm(samples, sample_rate)

        except Exception as e:
            logger.error(f"Error during Kokoro TTS: {e}")

    async def generate_audio(
        self, text: str, 
        language: Optional[str], 
        media_folder: str
    ) -> Optional[str]:
        """Generate wav from text and save to the media folder."""
        
        filename = f"{secrets.token_hex(16)}.mp3"
        file_path = os.path.join(media_folder, "audio", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            samples, sample_rate = await asyncio.to_thread(
                self._model.create,
                text,
                voice=self.voice,
                speed=1.0,
                lang=self._get_language(language),
            )
            
            segment_data = (np.array(samples) * np.iinfo(np.int16).max).astype(np.int16)
            audio_segment = AudioSegment(
                segment_data.tobytes(),    
                frame_rate=sample_rate,  
                sample_width=2,   
                channels=1
            )
                                                                                                              
            await asyncio.to_thread(
                audio_segment.export, 
                file_path, 
                format="mp3"
            )
            logger.debug(f"Speech saved to: {file_path}")
        
            return f"audio/{filename}"
        
        except Exception as e:
            logger.error(f"Error during Kokoro TTS: {e}")
            return None

    def stop_playback(self) -> None:
        """Stop the audio playback. Thread-safe."""

        self.audio_player.stop_playback()
