from typing import List, Optional, Tuple, Union

try:
    import ctranslate2
    from faster_whisper import WhisperModel
except ImportError:
    ctranslate2 = None
    WhisperModel = None

import numpy as np

from config import logger


class FWTranscriber:
    """
    Faster Whisper transcriber service.
    
    Attributes:
        language: The target language code (e.g., "en") or None for auto-detection.
        device: "cuda" or "cpu".
        model: The underlying WhisperModel instance.
    """

    def __init__(self, language: str = "en") -> None:
        if ctranslate2 is None or WhisperModel is None:
            raise ImportError(
                "faster-whisper is not installed. You must install the 'voice-local' extra: "
                "`uv pip install -e .[voice-local]` or `uv pip install faster-whisper`."
            )

        self.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model: Optional[WhisperModel] = None

        if language.upper() == "MULTILINGUAL":
            self.language = None
            self.model_name = "large-v3"
        else:
            self.language = language
            self.model_name = "distil-medium.en" if language == "en" else "large-v3"

    def init_model(self) -> None:
        """Lazy initialization of the WhisperModel."""
        if self.model is not None:
            return
    
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Faster Whisper model: {e}")
            raise
        
        logger.debug(f"Initialized FW model '{self.model_name}' on {self.device}.")

    def transcribe_audio(
        self,
        audio_clip: Union[np.ndarray, List[float]]
    ) -> Optional[Tuple[str, str]]:
        """Transcribe the given audio clip. Returns None on failure — never a
        (None, None) tuple, which is truthy and slips past the caller's guard."""

        if self.model is None:
            logger.error("Faster Whisper model is not initialized.")
            return None

        if not isinstance(audio_clip, (np.ndarray, list)):
            logger.error(f"Invalid audio format: {type(audio_clip)}")
            return None

        if isinstance(audio_clip, list):
            audio_clip = np.array(audio_clip, dtype=np.float32)

        try:
            segments, info = self.model.transcribe(
                audio_clip,
                language=self.language,
                vad_filter=True,
                vad_parameters=dict(threshold=0.3)
            )

            text = "".join(segment.text for segment in segments).strip()
            detected_lang = info.language[:2].lower() if info.language else self.language
            return (text, detected_lang)

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return None

    def close(self) -> None:
        """Explicitly release resources."""
        if self.model:
            self.model.model.unload_model()
            self.model = None

    def __del__(self) -> None:
        self.close()
