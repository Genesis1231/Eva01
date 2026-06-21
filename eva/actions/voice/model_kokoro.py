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
from pydub import AudioSegment
from typing import Optional

import numpy as np
import onnxruntime as ort
from config import logger

try:    
    from kokoro_onnx import Kokoro
except ImportError:
    Kokoro = None


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
        if Kokoro is None:
            raise ImportError(
                "kokoro-onnx is not installed. You must install the 'voice-local' extra: "
                "`uv pip install -e .[voice-local]` or `uv pip install kokoro-onnx`."
            )

        onnx_path = _MODEL_DIR / "kokoro-v1.0.onnx"
        voices_path = _MODEL_DIR / "voices-v1.0.bin"

        if not onnx_path.exists() or not voices_path.exists():
            raise FileNotFoundError(f"Kokoro model files not found in {_MODEL_DIR}. ")

        # kokoro-onnx's GPU auto-detect is broken (find_spec("onnxruntime-gpu")
        # always returns None), so build the session ourselves with CUDA
        self.voice = voice
        self.audio_player = AudioPlayer()
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._model = Kokoro.from_session(session, str(voices_path))

    def _get_language(self, language: Optional[str]) -> str:
        return _LANG_MAP.get(language or "en", "en-us") if language else "en-us"

    def synthesize(self, text: str, language: Optional[str] = None):
        """Render text to PCM without playing — speech is prepared ahead, then played
        when the floor is free. Returns (samples, sample_rate) or None on failure."""

        if not self._model:
            logger.error("KokoroSpeaker: TTS model not initialized.")
            return None

        try:
            return self._model.create(
                text=text,
                voice=self.voice,
                lang=self._get_language(language),
            )
        except Exception as e:
            logger.error(f"Error during Kokoro TTS synthesis: {e}")
            return None

    def play(self, prepared) -> None:
        """Play PCM prepared by synthesize(). Blocking — run via to_thread."""
        samples, sample_rate = prepared
        self.audio_player.play_pcm(samples, sample_rate)

    def eva_speak(self, text: str, language: Optional[str] = None) -> None:
        """Synthesize and play in one call. Blocking — run via to_thread."""

        prepared = self.synthesize(text, language)
        if prepared is not None:
            self.play(prepared)

    async def generate_audio(
        self, text: str, 
        language: Optional[str], 
        media_folder: str
    ) -> Optional[str]:
        """Generate wav from text and save to the media folder."""

        if not self._model:
            logger.error("KokoroSpeaker: TTS model not initialized.")
            return
    
        
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

    def close(self) -> None:
        """Release the ONNX session and voice data."""
        if hasattr(self, '_model') and self._model:
            self._model.sess._sess = None
            self._model = None
            logger.debug("KokoroSpeaker: ONNX session released.")
